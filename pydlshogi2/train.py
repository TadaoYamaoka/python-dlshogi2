import argparse
import logging
import torch
import torch.optim as optim

from pydlshogi2.network.policy_value_resnet import PolicyValueNetwork
from pydlshogi2.dataloader import HcpeDataLoader

parser = argparse.ArgumentParser(description='Train policy value network')
parser.add_argument('train_data', type=str, nargs='+', help='training data file')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
parser.add_argument('--testbatchsize', type=int, default=1024, help='Number of positions in each test mini-batch')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--checkpoint', default='checkpoints/checkpoint-{epoch:03}.pth', help='checkpoint file name')
parser.add_argument('--resume', '-r', default='', help='Resume from snapshot')
parser.add_argument('--eval_interval', type=int, default=100, help='evaluation interval')
parser.add_argument('--log', default=None, help='log file path')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)
logging.info('batchsize={}'.format(args.batchsize))
logging.info('lr={}'.format(args.lr))

# デバイス
if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

# モデル
model = PolicyValueNetwork()
model.to(device)

# オプティマイザ
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

# 損失関数
cross_entropy_loss = torch.nn.CrossEntropyLoss()
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

# チェックポイント読み込み
if args.resume:
    logging.info('Loading the checkpoint from {}'.format(args.resume))
    checkpoint = torch.load(args.resume, map_location=device)
    epoch = checkpoint['epoch']
    t = checkpoint['t']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # 学習率を引数の値に変更
    optimizer.param_groups[0]['lr'] = args.lr
else:
    epoch = 0
    t = 0  # total steps

# 訓練データ読み込み
logging.info('Reading training data')
train_dataloader = HcpeDataLoader(args.train_data, args.batchsize, device, shuffle=True)
# テストデータ読み込み
logging.info('Reading test data')
test_dataloader = HcpeDataLoader(args.test_data, args.testbatchsize, device)

# 読み込んだデータ数を表示
logging.info('train position num = {}'.format(len(train_dataloader)))
logging.info('test position num = {}'.format(len(test_dataloader)))

# 方策の正解率
def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

# 価値の正解率
def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)

# チェックポイント保存
def save_checkpoint():
    path = args.checkpoint.format(**{'epoch':epoch, 'step':t})
    logging.info('Saving the checkpoint to {}'.format(path))
    checkpoint = {
        'epoch': epoch,
        't': t,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


# 訓練ループ
for e in range(args.epoch):
    epoch += 1
    steps_interval = 0
    sum_loss_policy_interval = 0
    sum_loss_value_interval = 0
    steps_epoch = 0
    sum_loss_policy_epoch = 0
    sum_loss_value_epoch = 0
    for x, move_label, result in train_dataloader:
        model.train()

        # 順伝播
        y1, y2 = model(x)
        # 損失計算
        loss_policy = cross_entropy_loss(y1, move_label)
        loss_value = bce_with_logits_loss(y2, result)
        loss = loss_policy + loss_value
        # 誤差逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # トータルステップ数に加算
        t += 1

        # 評価間隔ごとのステップ数カウンタと損失合計に加算
        steps_interval += 1
        sum_loss_policy_interval += loss_policy.item()
        sum_loss_value_interval += loss_value.item()

        # 評価間隔ごとに訓練損失とテスト損失・正解率を表示
        if t % args.eval_interval == 0:
            model.eval()

            x, move_label, result = test_dataloader.sample()
            with torch.no_grad():
                # 推論
                y1, y2 = model(x)
                # 損失計算
                test_loss_policy = cross_entropy_loss(y1, move_label).item()
                test_loss_value = bce_with_logits_loss(y2, result).item()
                # 正解率計算
                test_accuracy_policy = accuracy(y1, move_label)
                test_accuracy_value = binary_accuracy(y2, result)

                # ログ表示
                logging.info('epoch = {}, steps = {}, train loss = {:.07f}, {:.07f}, {:.07f}, test loss = {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}'.format(
                    epoch, t,
                    sum_loss_policy_interval / steps_interval,
                    sum_loss_value_interval / steps_interval,
                    (sum_loss_policy_interval +
                     sum_loss_value_interval) / steps_interval,
                    test_loss_policy,
                    test_loss_value,
                    test_loss_policy + test_loss_value,
                    test_accuracy_policy,
                    test_accuracy_value))

            # エポックごとのステップ数カウンタと損失合計に加算
            steps_epoch += steps_interval
            sum_loss_policy_epoch += sum_loss_policy_interval
            sum_loss_value_epoch += sum_loss_value_interval

            # 評価間隔ごとのステップ数カウンタと損失合計をクリア
            steps_interval = 0
            sum_loss_policy_interval = 0
            sum_loss_value_interval = 0

    # エポックごとのステップ数カウンタと損失合計に加算
    steps_epoch += steps_interval
    sum_loss_policy_epoch += sum_loss_policy_interval
    sum_loss_value_epoch += sum_loss_value_interval

    # エポックの終わりにテストデータすべてを使用して評価する
    test_steps = 0
    sum_test_loss_policy = 0
    sum_test_loss_value = 0
    sum_test_accuracy_policy = 0
    sum_test_accuracy_value = 0
    model.eval()
    with torch.no_grad():
        for x, move_label, result in test_dataloader:
            y1, y2 = model(x)

            test_steps += 1
            sum_test_loss_policy += cross_entropy_loss(y1, move_label).item()
            sum_test_loss_value += bce_with_logits_loss(y2, result).item()
            sum_test_accuracy_policy += accuracy(y1, move_label)
            sum_test_accuracy_value += binary_accuracy(y2, result)

    # テストデータの検証結果をログ表示
    logging.info('epoch = {}, steps = {}, train loss avr = {:.07f}, {:.07f}, {:.07f}, test loss = {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}'.format(
        epoch, t,
        sum_loss_policy_epoch / steps_epoch,
        sum_loss_value_epoch / steps_epoch,
        (sum_loss_policy_epoch + sum_loss_value_epoch) / steps_epoch,
        sum_test_loss_policy / test_steps,
        sum_test_loss_value / test_steps,
        (sum_test_loss_policy + sum_test_loss_value) / test_steps,
        sum_test_accuracy_policy / test_steps,
        sum_test_accuracy_value / test_steps))

    # チェックポイント保存
    if args.checkpoint:
        save_checkpoint()
