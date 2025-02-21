class dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def bark(self):
        print(ani.name,"is barking")

class cat:
    def sleep(self):
        print(ani.name,"is sleeping")

class animal(dog,cat):
    def __init__(self, name, age, color):
        super().__init__(name,age)
        self.color=color
    def play(self):
        print(ani.name,"is playing")
ani=animal("jack",20,"black")
ani.bark()
ani.sleep()
ani.play()
print(ani.color)
    