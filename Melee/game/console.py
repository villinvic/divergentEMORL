from Melee.game.dolphin import DolphinInstance
from Melee.game.state import GameState
from Melee.input.pad import Pad
from Melee.player.player import Player, PlayerGroup
from Melee.characters.characters import *
from config.Loader import Default

from time import sleep, time
import os
import sys

class Console(Default):
    def __init__(self,
                 ID,
                 test,
                 ):
        super(Console, self).__init__()
        self.id = ID
        self.dolphin = DolphinInstance(self.exe_path, self.iso_path, test, ID)
        self.state = GameState(mw_path=self.mw_path, instance_id=ID, test=test)

        instance_pad_path = self.pad_path + str(ID) + '/'
        if not os.path.isdir(instance_pad_path):
            os.makedirs(instance_pad_path)

        self.state_dim = self.state.size
        self.action_dim = Pad.action_dim


    def close(self):
        self.dolphin.close()
        self.state.mw.unbind()

