class Dog:
    def __init__(self,name,age):
        self.name = name
        self.age = age

class Puppy(Dog):
    def __init__(self,name,age):
        super().__init__(name,age)
    def bark(self):
        print(pup.name,"is barking")
    def sleep(self):
        print(pup.name,"is sleeping")
    def play(self):
        print(self.name,"is playing")
pup=Puppy("jack",5)
pup.bark()
pup.sleep()
pup.play()

