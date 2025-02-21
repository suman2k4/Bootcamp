class Robot:
    def __init__(self,name,energy,model):
        self.name=name
        self._energy=energy
        self.__model=model

    def get_model(self):
        return self.__model
    
    def set_model(self,__model):
        if  model > 0:
            self.__model = model
        else:
            print("invalid")

bot=Robot("chitti",1200,3)
print(bot.name)
print(bot._energy)
print(bot.get_model())