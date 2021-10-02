import numpy as np

class UctNode:
    def __init__(self):
        self.move_count = 0           # ノードの訪問回数
        self.sum_value = 0.0          # 勝率の合計
        self.child_move = None        # 子ノードの指し手(リスト)
        self.child_move_count = None  # 子ノードの訪問回数(ndarray)
        self.child_sum_value = None   # 子ノードの勝率の合計(ndarray)
        self.child_node = None        # 子ノード(リスト)
        self.policy = None            # 方策ネットワークの予測確率(ndarray)
        self.value = None             # 価値

    # 子ノード作成
    def create_child_node(self, index):
        self.child_node[index] = UctNode()
        return self.child_node[index]

    # ノードの展開
    def expand_node(self, board):
        self.child_move = list(board.legal_moves)
        child_num = len(self.child_move)
        self.child_move_count = np.zeros(child_num, dtype=np.int32)
        self.child_sum_value = np.zeros(child_num, dtype=np.float32)

    # 1つを除くすべての子を削除する
    def release_children_except_one(self, move):
        if self.child_move and self.child_node:
            # 一つを残して削除する
            for i in range(len(self.child_move)):
                if self.child_move[i] == move:
                    if self.child_node[i] is None:
                        # 新しいノードを作成する
                        self.child_node[i] = UctNode()
                    # 子ノードを一つにする
                    if len(self.child_move) > 1:
                        self.child_move = [move]
                        self.child_move_count = None
                        self.child_sum_value = None
                        self.policy = None
                        self.child_node = [self.child_node[i]]
                    return self.child_node[0]

        # 子ノードが見つからなかった場合、または子ノードが未展開、または子ノードリストが未初期化の場合
        self.child_move = [move]
        self.child_move_count = None
        self.child_sum_value = None
        self.policy = None
        # 子ノードのリストを初期化する
        self.child_node = [UctNode()]
        return self.child_node[0]

class NodeTree:
    def __init__(self):
        self.current_head = None
        self.gamebegin_node = None
        self.history_starting_pos_key = None

    # ゲーム木内の位置を設定し、サブツリーの再利用を試みる
    def reset_to_position(self, starting_pos_key, moves):
        if self.history_starting_pos_key != starting_pos_key:
            # 開始位置が異なる場合、ゲーム木を作り直す
            self.gamebegin_node = UctNode()
            self.current_head = self.gamebegin_node

        self.history_starting_pos_key = starting_pos_key

        # 開始位置から順に、子ノード一つだけ残して、それ以外を解放する
        old_head = self.current_head
        prev_head = None
        self.current_head = self.gamebegin_node
        seen_old_head = self.gamebegin_node == old_head
        for move in moves:
            prev_head = self.current_head
            # current_headに着手を追加する
            self.current_head = self.current_head.release_children_except_one(move)
            if old_head == self.current_head:
                seen_old_head = True

        # 古いヘッドが現れない場合は、以前に探索された位置の祖先である位置がある可能性があることを意味する
        # つまり、古い子が以前にトリミングされていても、current_headは古いデータを保持する可能性がある
        # その場合、current_headをリセットする必要がある
        if not seen_old_head and self.current_head != old_head:
            if prev_head:
                assert len(prev_head.child_move) == 1
                prev_head.child_node[0] = UctNode()
                self.current_head = prev_head.child_node[0]
            else:
                # 開始局面に戻った場合
                self.gamebegin_node = UctNode()
                self.current_head = self.gamebegin_node
