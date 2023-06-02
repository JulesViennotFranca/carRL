import event
import machineActions

class Training():
    def __init__(self):
        super().__init__()

    def set_action(self, acc, turn):
        self.acc = acc 
        self.turn = turn

    def update(self):
        return self.acc, self.turn

class Human():
    def __init__(self):
        super().__init__()
    
    def update(self):
        return event.acceleration(), event.turn()

class Machine():
    def __init__(self, brain):
        super().__init__()
        self.brain = brain

    def set_game(self, game):
        self.game = game
    
    def update(self):
        obs = machineActions.observe(self.game)
        act = self.brain.act(obs)
        return machineActions.act(act)