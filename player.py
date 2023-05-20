import event
import machineActions
import config

class Player():
    def __init__(self):
        pass

    def update(self):
        pass

class TrainingPlayer(Player):
    def __init__(self):
        super().__init__()

    def set_action(self, acc, turn):
        self.acc = acc 
        self.turn = turn

    def update(self):
        return self.acc, self.turn

class HumanPlayer(Player):
    def __init__(self):
        super().__init__()
    
    def update(self):
        return event.acceleration(), event.turn()

class MachinePlayer(Player):
    def __init__(self, engine):
        super(MachinePlayer, self).__init__()
        self.engine = engine
        self.action_counter = config.machine_action_spand

    def set_game(self, game):
        self.game = game
    
    def update(self):
        if self.action_counter == config.machine_action_spand:
            obs = machineActions.observe(self.game)
            self.act = self.engine.act(obs)
            self.action_counter = 0
        self.action_counter += 1
        return machineActions.act(self.act)