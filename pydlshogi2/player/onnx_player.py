import onnxruntime
import numpy as np

from pydlshogi2.player.mcts_player import MCTSPlayer
from cshogi.dlshogi import make_input_features, make_move_label, FEATURES1_NUM, FEATURES2_NUM

class OnnxPlayer(MCTSPlayer):
    # USIエンジンの名前
    name = 'python-dlshogi-onnx'
    # デフォルトモデル
    DEFAULT_MODELFILE = 'model/model-0000167.onnx'

    # モデルのロード
    def load_model(self):
        self.session = onnxruntime.InferenceSession(self.modelfile, providers=['CUDAExecutionProvider'])

    # 入力特徴量の初期化
    def init_features(self):
        self.features = [
            np.empty((self.batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32),
            np.empty((self.batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32)
        ]

    # 入力特徴量の作成
    def make_input_features(self, board):
        make_input_features(board,
            self.features[0][self.current_batch_index],
            self.features[1][self.current_batch_index])

    # 推論
    def infer(self):
        io_binding = self.session.io_binding()
        io_binding.bind_cpu_input('input1', self.features[0][0:self.current_batch_index])
        io_binding.bind_cpu_input('input2', self.features[1][0:self.current_batch_index])
        io_binding.bind_output('output_policy')
        io_binding.bind_output('output_value')
        self.session.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()

    # 着手を表すラベル作成
    def make_move_label(self, move, color):
        return make_move_label(move, color)

if __name__ == '__main__':
    player = OnnxPlayer()
    player.run()
