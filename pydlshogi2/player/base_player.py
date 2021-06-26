class BasePlayer:
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

    def go(self, btime=None, wtime=None, byoyomi=None, binc=None, winc=None, nodes=None):
        pass

    def quit(self):
        pass

    def run(self):
        while True:
            cmd_line = input().strip()
            cmd = cmd_line.split(' ', 1)

            if cmd[0] == 'usi':
                self.usi()
            elif cmd[0] == 'setoption':
                option = cmd[1].split(' ')
                self.setoption(option)
            elif cmd[0] == 'isready':
                self.isready()
            elif cmd[0] == 'usinewgame':
                self.usinewgame()
            elif cmd[0] == 'position':
                moves = cmd[1].split(' ')
                self.position(moves)
            elif cmd[0] == 'go':
                kwargs = {}
                if len(cmd) > 1:
                    args = cmd[1].split(' ')
                    for i in range(0, len(args) - 1, 2):
                        if args[i] in ['btime', 'wtime', 'byoyomi', 'binc', 'winc', 'nodes']:
                            kwargs[args[i]] = int(args[i + 1])
                self.go(**kwargs)
            elif cmd[0] == 'quit':
                self.quit()
                break
