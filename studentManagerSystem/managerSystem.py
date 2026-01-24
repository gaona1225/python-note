# 1、存储数据的位置：文件（student.data）
#   加载文件数据
#   修改数据后保存的文件
# 2、存储数据的形式：列表存储学员对象
# 3、系统功能
#   添加学员
#   删除学员
#   修改学员
#   查询学员
#   显示所有学员信息
#   保存学员信息
#   退出系统


from student import *

class StudentManager(object):
    def __init__(self):
        # 存储数据所用的列表
        self.student_list = []
    
    # 定义程序入口函数--启动程序后执行的函数
    def run(self):
        # 1.加载数据--加载学员信息
        self.load_student()
        while True:
            # 2.显示功能菜单
            self.show_menu()
            # 3.用户输入功能序号
            menu_num = int(input('请输入您需要的功能序号:'))
            # 4.根据用户输入的功能序号执行不同的功能
            if menu_num == 1:
                # 1:添加学员
                self.add_student()
            elif menu_num == 2:
                # 2:删除学员
                self.del_student()
            elif menu_num == 3:
                # 3:修改学员
                self.modify_student()
            elif menu_num == 4:
                # 4:查询学员
                self.search_student()
            elif menu_num == 5:
                # 5:显示所有学员信息
                self.show_student()
            elif menu_num == 6:
                # 6:保存学员信息
                self.save_student()
            elif menu_num == 7:
                # 7:退出系统
                break
    # 定义系统功能函数，添加、删除学员等
    # 显示功能菜单--打印序号的功能对应关系--定义成静态方法
    @staticmethod
    def show_menu():
        print('=======')
        print('请选择如下功能:')
        print('1:添加学员信息')
        print('2:删除学员信息')
        print('3:修改学员信息')
        print('4:查询学员信息')
        print('5:显示所有学员信息')
        print('6:保存学员信息')
        print('7:退出系统')
        print('=======')

    # 添加学员信息
    def add_student(self):
        # 用户输入学员姓名、性别、手机号，并将学员添加到系统 
        name = input('请输入您的姓名:')
        gender = input('请输入您的性别:')
        tel = input('请输入您的手机号:')
        # 创建学员对象
        for stu in self.student_list:
            print(f'===name 是{name}, stu.name 是{stu.name}')
            if name == stu.name:
                print(f'已存在学员{name}')
                return
        student = Student(name, gender, tel)
        # 将该对象添加到学员列表
        self.student_list.append(student)
        print(f'学员列表：{self.student_list}')
        print(student)

    # 删除学员信息
    def del_student(self):
        # 输入待删除的学员姓名
        del_name = input('请输入要删除的学员姓名：')
        # 如果输入的学员存在就删除，不存在就提示
        for stu in self.student_list:
            if del_name == stu.name:
                self.student_list.remove(stu)
                break
        else:
            print('查无此人')
        print(f'学员列表：{self.student_list}')

    # 修改学员信息
    def modify_student(self):
        # 输入待修改的学员姓名
        modify_name = input('请输入需要修改的学员姓名：')
        # 如果输入的学员存在则修改，如果不存在给个提示
        for stu in self.student_list:
            if modify_name == stu.name:
                stu.name = input('姓名：')
                stu.gender = input('性别：')
                stu.tel = input('电话：')
                print(f'学员{modify_name}修改信息成功，姓名:{stu.name}, 性别为{stu.gender}, 电话为:{stu.tel}')
                break
        else:
            print('查无此人')

    # 查询学员信息
    def search_student(self):
        # 输入待查询的学员姓名
        search_name = input('请输入需要查询的学员姓名：')
        for stu in self.student_list:
            if search_name == stu.name:
                print(f'姓名:{stu.name}, 性别为{stu.gender}, 电话为:{stu.tel}')
                break
        else:
            print('查无此人')

    # 显示所有学员信息
    def show_student(self):
        print('姓名\t性别\t手机号')
        for stu in self.student_list:
            print(f'{stu.name}\t{stu.gender}\t{stu.tel}')
    
    # 保存学员信息
    def save_student(self):
        # 打开文件
        f = open('./study-AI/AIProjects/python-note/studentManagerSystem/student.data', 'w')
        # 文件写入数据
        # 1.文件写入的数据不能是学员对象的内存地址，需要把学员数据转换为列表字典数据再做存储
        new_list = [i.__dict__ for i in self.student_list]
        # 2.文件内数据要求为字符串类型，故需要先转换数据类型为字符串才能在文件中写入数据
        f.write(str(new_list))
        # 关闭文件
        f.close()

    # 加载学员信息
    def load_student(self):
        # 每次进入系统后，修改的数据是文件里的数据
        # 尝试以'r'模式打开学员数据文件，如果文件不存在则以'w'模式打开文件
        try:
            f = open('./study-AI/AIProjects/python-note/studentManagerSystem/student.data', 'r')
        except:
            f = open('./study-AI/AIProjects/python-note/studentManagerSystem/student.data', 'w')
        else:
            # 如果文件存在则读取数据并存储数据
            # 1.读取文件
            data = f.read()
            # 2.转换数据类型为列表并转换列表内的字典为对象
            new_list = eval(data)
            # 3.存储学员数据到学员列表
            self.student_list = [Student(s['name'], s['gender'], s['tel']) for s in new_list]
        finally:
            # 关闭文件
            f.close()