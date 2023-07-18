import event
import machineActions
import config

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
        self.action_counter = config.machine_action_spand

    def set_game(self, game):
        self.game = game

    def reset(self):
        self.action_counter = config.machine_action_spand
    
    def update(self):
        if self.action_counter == config.machine_action_spand:
            obs = machineActions.observe(self.game)
            self.act = self.brain.act(obs)
            self.action_counter = 0
        self.action_counter += 1
        return machineActions.act(self.act)