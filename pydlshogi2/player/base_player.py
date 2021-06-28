import time
from concurrent.futures import ThreadPoolExecutor

class BasePlayer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def usi(self):
        pass

    def usinewgame(self):
        pass

    def setoption(self, args):
        pass

    def isready(self):
        pass

    def position(self, args):
        pass

    def go(self, btime=None, wtime=None, byoyomi=None, binc=None, winc=None, nodes=None, infinite=False, ponder=False):
        pass

    def stop(self):
        pass

    def ponderhit(self, last_condition, elapsed):
        pass

    def quit(self):
        pass

    def run(self):
        while True:
            cmd_line = input().strip()
            cmd = cmd_line.split(' ', 1)

            if cmd[0] == 'usi':
                self.usi()
                print('usiok', flush=True)
            elif cmd[0] == 'setoption':
                option = cmd[1].split(' ')
                self.setoption(option)
            elif cmd[0] == 'isready':
                self.isready()
                print('readyok', flush=True)
            elif cmd[0] == 'usinewgame':
                self.usinewgame()
            elif cmd[0] == 'position':
                moves = cmd[1].split(' ')
                self.position(moves)
            elif cmd[0] == 'go':
                kwargs = {}
                if len(cmd) > 1:
                    args = cmd[1].split(' ')
                    if args[0] == 'infinite':
                        kwargs['infinite'] = True
                    elif args[0] == 'ponder':
                        kwargs['ponder'] = True
                    else:
                        for i in range(0, len(args) - 1, 2):
                            if args[i] in ['btime', 'wtime', 'byoyomi', 'binc', 'winc', 'nodes']:
                                kwargs[args[i]] = int(args[i + 1])
                start_time = time.time()
                self.future = self.executor.submit(self.go, **kwargs)
                if 'ponder' in kwargs:
                    # ponderhitのために経過時間を加算
                    elapsed += int((time.time() - start_time) *  1000)
                elif 'infinite' not in kwargs:
                    bestmove, ponder_move = self.future.result()
                    print('bestmove ' + bestmove + (' ponder ' + ponder_move if ponder_move else ''), flush=True)
                    # ponderhitのために条件と経過時間を保存
                    last_condition = kwargs
                    elapsed = int((time.time() - start_time) *  1000)
            elif cmd[0] == 'stop':
                self.stop()
                bestmove, _ = self.future.result()
                print('bestmove ' + bestmove, flush=True)
            elif cmd[0] == 'ponderhit':
                start_time = time.time()
                self.ponderhit(last_condition, elapsed)
                bestmove, ponder_move = self.future.result()
                print('bestmove ' + bestmove + (' ponder ' + ponder_move if ponder_move else ''), flush=True)
            elif cmd[0] == 'quit':
                self.quit()
                break
