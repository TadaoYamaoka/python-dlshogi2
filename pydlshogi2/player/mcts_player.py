import numpy as np
import torch

from cshogi import Board, BLACK, NOT_REPETITION, REPETITION_DRAW, REPETITION_WIN, REPETITION_SUPERIOR, move_to_usi
from pydlshogi2.features import FEATURES_NUM, make_input_features, make_move_label
from pydlshogi2.uct.uct_node import NodeTree
from pydlshogi2.network.policy_value_resnet import PolicyValueNetwork
from pydlshogi2.player.base_player import BasePlayer

import time
import math

# デフォルトGPU ID
DEFAULT_GPU_ID = 0
# デフォルトバッチサイズ
DEFAULT_BATCH_SIZE = 32
# デフォルト投了閾値
DEFAULT_RESIGN_THRESHOLD = 0.01
# デフォルトPUCTの定数
DEFAULT_C_PUCT = 1.0
# デフォルト温度パラメータ
DEFAULT_TEMPERATURE = 1.0
# デフォルト持ち時間マージン(ms)
DEFAULT_TIME_MARGIN = 1000
# デフォルト秒読みマージン(ms)
DEFAULT_BYOYOMI_MARGIN = 100
# デフォルトPV表示間隔(ms)
DEFAULT_PV_INTERVAL = 500
# デフォルトプレイアウト数
DEFAULT_CONST_PLAYOUT = 1000
# 勝ちを表す定数（数値に意味はない）
VALUE_WIN = 10000
# 負けを表す定数（数値に意味はない）
VALUE_LOSE = -10000
# 引き分けを表す定数（数値に意味はない）
VALUE_DRAW = 20000
# キューに追加されたときの戻り値（数値に意味はない）
QUEUING = -1
# 探索を破棄するときの戻り値（数値に意味はない）
DISCARDED = -2
# Virtual Loss
VIRTUAL_LOSS = 1

# 温度パラメータを適用した確率分布を取得
def softmax_temperature_with_normalize(logits, temperature):
    # 温度パラメータを適用
    logits /= temperature

    # 確率を計算(オーバーフローを防止するため最大値で引く)
    max_logit = max(logits)
    probabilities = np.exp(logits - max_logit)

    # 合計が1になるように正規化
    sum_probabilities = sum(probabilities)
    probabilities /= sum_probabilities

    return probabilities

# ノード更新
def update_result(current_node, next_index, result):
    current_node.sum_value += result
    current_node.move_count += 1 - VIRTUAL_LOSS
    current_node.child_sum_value[next_index] += result
    current_node.child_move_count[next_index] += 1 - VIRTUAL_LOSS

# 評価待ちキューの要素
class EvalQueueElement:
    def set(self, node, color):
        self.node = node
        self.color = color

class MCTSPlayer(BasePlayer):
    # USIエンジンの名前
    name = 'python-dlshogi2'
    # デフォルトチェックポイント
    DEFAULT_MODELFILE = 'checkpoints/checkpoint.pth'

    def __init__(self):
        super().__init__()
        # チェックポイントのパス
        self.modelfile = self.DEFAULT_MODELFILE
        # モデル
        self.model = None
        # 入力特徴量
        self.features = None
        # 評価待ちキュー
        self.eval_queue = None
        # バッチインデックス
        self.current_batch_index = 0

        # ルート局面
        self.root_board = Board()
        # ゲーム木
        self.tree = NodeTree()

        # プレイアウト回数
        self.playout_count = 0
        # 中断するプレイアウト回数
        self.halt = None

        # GPU ID
        self.gpu_id = DEFAULT_GPU_ID
        # デバイス
        self.device = None
        # バッチサイズ
        self.batch_size = DEFAULT_BATCH_SIZE

        # 投了する勝率の閾値
        self.resign_threshold = DEFAULT_RESIGN_THRESHOLD
        # PUCTの定数
        self.c_puct = DEFAULT_C_PUCT
        # 温度パラメータ
        self.temperature = DEFAULT_TEMPERATURE
        # 持ち時間マージン(ms)
        self.time_margin = DEFAULT_TIME_MARGIN
        # 秒読みマージン(ms)
        self.byoyomi_margin = DEFAULT_BYOYOMI_MARGIN
        # PV表示間隔
        self.pv_interval = DEFAULT_PV_INTERVAL

        self.debug = False

    def usi(self):
        print('id name ' + self.name)
        print('option name USI_Ponder type check default false')
        print('option name modelfile type string default ' + self.DEFAULT_MODELFILE)
        print('option name gpu_id type spin default ' + str(DEFAULT_GPU_ID) + ' min -1 max 7')
        print('option name batchsize type spin default ' + str(DEFAULT_BATCH_SIZE) + ' min 1 max 256')
        print('option name resign_threshold type spin default ' + str(int(DEFAULT_RESIGN_THRESHOLD * 100)) + ' min 0 max 100')
        print('option name c_puct type spin default ' + str(int(DEFAULT_C_PUCT * 100)) + ' min 10 max 1000')
        print('option name temperature type spin default ' + str(int(DEFAULT_TEMPERATURE * 100)) + ' min 10 max 1000')
        print('option name time_margin type spin default ' + str(DEFAULT_TIME_MARGIN) + ' min 0 max 1000')
        print('option name byoyomi_margin type spin default ' + str(DEFAULT_BYOYOMI_MARGIN) + ' min 0 max 1000')
        print('option name pv_interval type spin default ' + str(DEFAULT_PV_INTERVAL) + ' min 0 max 10000')
        print('option name debug type check default false')

    def setoption(self, args):
        if args[1] == 'modelfile':
            self.modelfile = args[3]
        elif args[1] == 'gpu_id':
            self.gpu_id = int(args[3])
        elif args[1] == 'batchsize':
            self.batch_size = int(args[3])
        elif args[1] == 'resign_threshold':
            self.resign_threshold = int(args[3]) / 100
        elif args[1] == 'c_puct':
            self.c_puct = int(args[3]) / 100
        elif args[1] == 'temperature':
            self.temperature = int(args[3]) / 100
        elif args[1] == 'time_margin':
            self.time_margin = int(args[3])
        elif args[1] == 'byoyomi_margin':
            self.byoyomi_margin = int(args[3])
        elif args[1] == 'pv_interval':
            self.pv_interval = int(args[3])
        elif args[1] == 'debug':
            self.debug = args[3] == 'true'

    # モデルのロード
    def load_model(self):
        self.model = PolicyValueNetwork()
        self.model.to(self.device)
        checkpoint = torch.load(self.modelfile, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        # モデルを評価モードにする
        self.model.eval()

    # 入力特徴量の初期化
    def init_features(self):
        self.features = torch.empty((self.batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=(self.gpu_id >= 0))

    def isready(self):
        # デバイス
        if self.gpu_id >= 0:
            self.device = torch.device(f"cuda:{self.gpu_id}")
        else:
            self.device = torch.device("cpu")

        # モデルをロード
        self.load_model()

        # 局面初期化
        self.root_board.reset()
        self.tree.reset_to_position(self.root_board.zobrist_hash(), [])

        # 入力特徴量と評価待ちキューを初期化
        self.init_features()
        self.eval_queue = [EvalQueueElement() for _ in range(self.batch_size)]
        self.current_batch_index = 0

        # モデルをキャッシュして初回推論を速くする
        current_node = self.tree.current_head
        current_node.expand_node(self.root_board)
        for _ in range(self.batch_size):
            self.queue_node(self.root_board, current_node)
        self.eval_node()

    def position(self, sfen, usi_moves):
        if sfen == 'startpos':
            self.root_board.reset()
        elif sfen[:5] == 'sfen ':
            self.root_board.set_sfen(sfen[5:])

        starting_pos_key = self.root_board.zobrist_hash()

        moves = []
        for usi_move in usi_moves:
            move = self.root_board.push_usi(usi_move)
            moves.append(move)
        self.tree.reset_to_position(starting_pos_key, moves)

        if self.debug:
            print(self.root_board)

    def set_limits(self, btime=None, wtime=None, byoyomi=None, binc=None, winc=None, nodes=None, infinite=False, ponder=False):
        # 探索回数の閾値を設定
        if infinite or ponder:
            # infiniteもしくはponderの場合は、探索を打ち切らないため、32ビット整数の最大値を設定する
            self.halt = 2**31-1
        elif nodes:
            # プレイアウト数固定
            self.halt = nodes
        else:
            self.remaining_time, inc = (btime, binc) if self.root_board.turn == BLACK else (wtime, winc)
            if self.remaining_time is None and byoyomi is None and inc is None:
                # 時間指定がない場合
                self.halt = DEFAULT_CONST_PLAYOUT
            else:
                self.minimum_time = 0
                self.remaining_time = int(self.remaining_time) if self.remaining_time else 0
                inc = int(inc) if inc else 0
                self.time_limit = self.remaining_time / (14 + max(0, 30 - self.root_board.move_number)) + inc
                # 秒読みの場合
                if byoyomi:
                    byoyomi = int(byoyomi) - self.byoyomi_margin
                    self.minimum_time = byoyomi
                    # time_limitが秒読み以下の場合、秒読みに設定
                    if self.time_limit < byoyomi:
                        self.time_limit = byoyomi
                self.extend_time = self.time_limit > self.minimum_time
                self.halt = None

    def go(self):
        # 探索開始時刻の記録
        self.begin_time = time.time()

        # 投了チェック
        if self.root_board.is_game_over():
            return 'resign', None

        # 入玉宣言勝ちチェック
        if self.root_board.is_nyugyoku():
            return 'win', None

        current_node = self.tree.current_head

        # 詰みの場合
        if current_node.value == VALUE_WIN:
            matemove = self.root_board.mate_move(3)
            if matemove != 0:
                print('info score mate 3 pv {}'.format(move_to_usi(matemove)), flush=True)
                return move_to_usi(matemove), None
        if not self.root_board.is_check():
            matemove = self.root_board.mate_move_in_1ply()
            if matemove:
                print('info score mate 1 pv {}'.format(move_to_usi(matemove)), flush=True)
                return move_to_usi(matemove), None

        # プレイアウト数をクリア
        self.playout_count = 0

        # ルートノードが未展開の場合、展開する
        if current_node.child_move is None:
            current_node.expand_node(self.root_board)

        # 候補手が1つの場合は、その手を返す
        if self.halt is None and len(current_node.child_move) == 1:
            if current_node.child_move_count[0] > 0:
                bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()
                return move_to_usi(bestmove), move_to_usi(ponder_move) if ponder_move else None
            else:
                return move_to_usi(current_node.child_move[0]), None

        # ルートノードが未評価の場合、評価する
        if current_node.policy is None:
            self.current_batch_index = 0
            self.queue_node(self.root_board, current_node)
            self.eval_node()

        # 探索
        self.search()

        # 最善手の取得とPVの表示
        bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()

        # for debug
        if self.debug:
            for i in range(len(current_node.child_move)):
                print('{:3}:{:5} move_count:{:4} nn_rate:{:.5f} win_rate:{:.5f}'.format(
                    i, move_to_usi(current_node.child_move[i]),
                    current_node.child_move_count[i],
                    current_node.policy[i],
                    current_node.child_sum_value[i] / current_node.child_move_count[i] if current_node.child_move_count[i] > 0 else 0))

        # 閾値未満の場合投了
        if bestvalue < self.resign_threshold:
            return 'resign', None

        return move_to_usi(bestmove), move_to_usi(ponder_move) if ponder_move else None

    def stop(self):
        # すぐに中断する
        self.halt = 0

    def ponderhit(self, last_limits):
        # 探索開始時刻の記録
        self.begin_time = time.time()
        self.last_pv_print_time = 0

        # プレイアウト数をクリア
        self.playout_count = 0

        # 探索回数の閾値を設定
        self.set_limits(**last_limits)

    def quit(self):
        self.stop()

    def search(self):
        self.last_pv_print_time = 0

        # 探索経路のバッチ
        trajectories_batch = []
        trajectories_batch_discarded = []

        # 探索回数が閾値を超える、または探索が打ち切られたらループを抜ける
        while True:
            trajectories_batch.clear()
            trajectories_batch_discarded.clear()
            self.current_batch_index = 0

            # バッチサイズの回数だけシミュレーションを行う
            for i in range(self.batch_size):
                # 盤面のコピー
                board = self.root_board.copy()

                # 探索
                trajectories_batch.append([])
                result = self.uct_search(board, self.tree.current_head, trajectories_batch[-1])

                if result != DISCARDED:
                    # 探索回数を1回増やす
                    self.playout_count += 1
                else:
                    # 破棄した探索経路を保存
                    trajectories_batch_discarded.append(trajectories_batch[-1])
                    # 破棄が多い場合はすぐに評価する
                    if len(trajectories_batch_discarded) > self.batch_size // 2:
                        trajectories_batch.pop()
                        break

                # 評価中の葉ノードに達した、もしくはバックアップ済みため破棄する
                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()

            # 評価
            if len(trajectories_batch) > 0:
                self.eval_node()

            # 破棄した探索経路のVirtual Lossを戻す
            for trajectories in trajectories_batch_discarded:
                for i in range(len(trajectories)):
                    current_node, next_index = trajectories[i]
                    current_node.move_count -= VIRTUAL_LOSS
                    current_node.child_move_count[next_index] -= VIRTUAL_LOSS

            # バックアップ
            for trajectories in trajectories_batch:
                result = None
                for i in reversed(range(len(trajectories))):
                    current_node, next_index = trajectories[i]
                    if result is None:
                        # 葉ノード
                        result = 1.0 - current_node.child_node[next_index].value
                    update_result(current_node, next_index, result)
                    result = 1.0 - result

            # 探索を打ち切るか確認
            if self.check_interruption():
                return

            # PV表示
            if self.pv_interval > 0:
                elapsed_time = int((time.time() - self.begin_time) * 1000)
                if elapsed_time > self.last_pv_print_time + self.pv_interval:
                    self.last_pv_print_time = elapsed_time
                    self.get_bestmove_and_print_pv()

    # UCT探索
    def uct_search(self, board, current_node, trajectories):
        # 子ノードのリストが初期化されていない場合、初期化する
        if not current_node.child_node:
            current_node.child_node = [None for _ in range(len(current_node.child_move))]
        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node)
        # 選んだ手を着手
        board.push(current_node.child_move[next_index])

        # Virtual Lossを加算
        current_node.move_count += VIRTUAL_LOSS
        current_node.child_move_count[next_index] += VIRTUAL_LOSS

        # 経路を記録
        trajectories.append((current_node, next_index))

        # ノードの展開の確認
        if current_node.child_node[next_index] is None:
            # ノードの作成
            child_node = current_node.create_child_node(next_index)

            # 千日手チェック
            draw = board.is_draw()
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    # 千日手
                    child_node.value = VALUE_DRAW
                    result = 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    # 連続王手の千日手で勝ち、もしくは優越局面の場合
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:  # draw == REPETITION_LOSE or draw == REPETITION_INFERIOR
                    # 連続王手の千日手で負け、もしくは劣等局面の場合
                    child_node.value = VALUE_LOSE
                    result = 1.0
            else:
                # 入玉宣言と3手詰めチェック
                if board.is_nyugyoku() or board.mate_move(3):
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:
                    # 候補手を展開する
                    child_node.expand_node(board)
                    # 候補手がない場合
                    if len(child_node.child_move) == 0:
                        child_node.value = VALUE_LOSE
                        result = 1.0
                    else:
                        # ノードを評価待ちキューに追加
                        self.queue_node(board, child_node)
                        return QUEUING
        else:
            # 評価待ちのため破棄する
            next_node = current_node.child_node[next_index]
            if next_node.value is None:
                return DISCARDED

            # 詰みと千日手チェック
            if next_node.value == VALUE_WIN:
                result = 0.0
            elif next_node.value == VALUE_LOSE:
                result = 1.0
            elif next_node.value == VALUE_DRAW:
                result = 0.5
            elif len(next_node.child_move) == 0:
                result = 1.0
            else:
                # 手番を入れ替えて1手深く読む
                result = self.uct_search(board, next_node, trajectories)

        if result == QUEUING or result == DISCARDED:
            return result

        # 探索結果の反映
        update_result(current_node, next_index, result)

        return 1.0 - result

    # UCB値が最大の手を求める
    def select_max_ucb_child(self, node):
        q = np.divide(node.child_sum_value, node.child_move_count,
            out=np.zeros(len(node.child_move), np.float32),
            where=node.child_move_count != 0)
        if node.move_count == 0:
            u = 1.0
        else:
            u = np.sqrt(np.float32(node.move_count)) / (1 + node.child_move_count)
        ucb = q + self.c_puct * u * node.policy

        return np.argmax(ucb)

    # 最善手取得とinfoの表示
    def get_bestmove_and_print_pv(self):
        # 探索にかかった時間を求める
        finish_time = time.time() - self.begin_time

        # 訪問回数最大の手を選択する
        current_node = self.tree.current_head
        selected_index = np.argmax(current_node.child_move_count)

        # 選択した着手の勝率の算出
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]

        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)

        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)

        return bestmove, bestvalue, ponder_move

    # 探索を打ち切るか確認
    def check_interruption(self):
        # プレイアウト数数が閾値を超えている
        if self.halt is not None:
            return self.playout_count >= self.halt

        # 候補手が1つの場合、中断する
        current_node = self.tree.current_head
        if len(current_node.child_move) == 1:
            return True

        # 消費時間
        spend_time = int((time.time() - self.begin_time) * 1000)

        # 消費時間が短すぎる場合、もしくは秒読みの場合は打ち切らない
        if spend_time * 10 < self.time_limit or spend_time < self.minimum_time:
            return False

        # 探索回数が最も多い手と次に多い手を求める
        child_move_count = current_node.child_move_count
        second_index, first_index = np.argpartition(child_move_count, -2)[-2:]
        second, first = child_move_count[[second_index, first_index]]

        # 探索速度から残りの時間で探索できるプレイアウト数を見積もる
        rest = int(self.playout_count * ((self.time_limit - spend_time) / spend_time))

        # 残りの探索で次善手が最善手を超える可能性がある場合は打ち切らない
        if first - second <= rest:
            return False

        # 探索延長
        #   21手目以降かつ、残り時間がある場合、
        #   最善手の探索回数が次善手の探索回数の1.5倍未満
        #   もしくは、勝率が逆なら探索延長する
        if self.extend_time and \
           self.root_board.move_number > 20 and \
           self.remaining_time > self.time_limit * 2 and \
           (first < second * 1.5 or
            current_node.child_sum_value[first_index] / child_move_count[first_index] < current_node.child_sum_value[second_index] / child_move_count[second_index]):
            # 探索時間を2倍に延長
            self.time_limit *= 2
            # 探索延長は1回のみ
            self.extend_time = False
            print('info string extend_time')
            return False

        return True

    # 入力特徴量の作成
    def make_input_features(self, board):
        make_input_features(board, self.features.numpy()[self.current_batch_index])

    # ノードをキューに追加
    def queue_node(self, board, node):
        # 入力特徴量を作成
        self.make_input_features(board)

        # ノードをキューに追加
        self.eval_queue[self.current_batch_index].set(node, board.turn)
        self.current_batch_index += 1

    # 推論
    def infer(self):
        with torch.no_grad():
            x = self.features[0:self.current_batch_index].to(self.device)
            policy_logits, value_logits = self.model(x)
            return policy_logits.cpu().numpy(), torch.sigmoid(value_logits).cpu().numpy()

    # 着手を表すラベル作成
    def make_move_label(self, move, color):
        return make_move_label(move, color)

    # 局面の評価
    def eval_node(self):
        # 推論
        policy_logits, values = self.infer()

        for i, (policy_logit, value) in enumerate(zip(policy_logits, values)):
            current_node = self.eval_queue[i].node
            color = self.eval_queue[i].color

            # 合法手一覧
            legal_move_probabilities = np.empty(len(current_node.child_move), dtype=np.float32)
            for j in range(len(current_node.child_move)):
                move = current_node.child_move[j]
                move_label = self.make_move_label(move, color)
                legal_move_probabilities[j] = policy_logit[move_label]

            # Boltzmann分布
            probabilities = softmax_temperature_with_normalize(legal_move_probabilities, self.temperature)

            # ノードの値を更新
            current_node.policy = probabilities
            current_node.value = float(value)

if __name__ == '__main__':
    player = MCTSPlayer()
    player.run()
