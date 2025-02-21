class Parent1:
    def show(self):
        print("Parent1")

class Parent2:
    def disp(self):
        print("Parent2")

class Child(Parent1,Parent2):
    def show(self):
        print("Child")

obj=Child()
obj.show()
obj.disp()

    
        