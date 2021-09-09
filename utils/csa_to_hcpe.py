import argparse
from cshogi import HuffmanCodedPosAndEval, Board, BLACK, move16
from cshogi import CSA
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('hcpe_train')
parser.add_argument('hcpe_test')
parser.add_argument('--filter_moves', type=int, default=50)
parser.add_argument('--filter_rating', type=int, default=3500)
parser.add_argument('--test_ratio', type=float, default=0.1)
args = parser.parse_args()

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True)

# ファイルリストをシャッフル
file_list_train, file_list_test = train_test_split(csa_file_list, test_size=args.test_ratio)

hcpes = np.zeros(1024, HuffmanCodedPosAndEval)

f_train = open(args.hcpe_train, 'wb')
f_test = open(args.hcpe_test, 'wb')

board = Board()
for file_list, f in zip([file_list_train, file_list_test], [f_train, f_test]):
    kif_num = 0
    position_num = 0
    for filepath in file_list:
        for kif in CSA.Parser.parse_file(filepath):
            # 投了、千日手、宣言勝ちで終了した棋譜以外を除外
            if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI'):
                continue
            # 手数が少ない棋譜を除外
            if len(kif.moves) < args.filter_moves:
                continue
            # レーティングの低いエンジンの対局を除外
            if args.filter_rating > 0 and mint(kif.ratings) < args.filter_rating:
                continue

            # 開始局面を設定
            board.set_sfen(kif.sfen)
            p = 0
            try:
                for i, (move, score, comment) in enumerate(zip(kif.moves, kif.scores, kif.comments)):
                    # 不正な指し手のある棋譜を除外
                    if not board.is_legal(move):
                        raise Exception()
                    hcpe = hcpes[p]
                    p += 1
                    # 局面はhcpに変換
                    board.to_hcp(hcpe['hcp'])
                    # 16bitに収まるようにクリッピングする
                    eval = min(32767, max(score, -32767))
                    # 手番側の評価値にする
                    hcpe['eval'] = eval if board.turn == BLACK else -eval
                    # 指し手の32bit数値を16bitに切り捨てる
                    hcpe['bestMove16'] = move16(move)
                    # 勝敗結果
                    hcpe['gameResult'] = kif.win
                    board.push(move)
            except:
                print(f'skip {filepath}')
                continue

            if p == 0:
                continue

            hcpes[:p].tofile(f)

            kif_num += 1
            position_num += p

    print('kif_num', kif_num)
    print('position_num', position_num)
