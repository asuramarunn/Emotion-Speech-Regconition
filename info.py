import os

def name():
    '''Hàm này giúp lấy tên các thành viên trong nhóm từ tên user trong máy'''
    user = os.getlogin()
    if user == 'nguye_ax':
        return 'Nguyễn Xuân An'
    elif user == 'Dell':
        return 'Võ Đình Đại'
    elif user == 'Dat':
        return 'Nguyễn Tiến Đạt'
    elif user == 'HP':
        return 'Trần Ngọc Phúc'
    elif user == 'Thang':
        return 'Nguyễn Hữu Thắng'
    else:
        return user


def member(name):
    '''Hàm này giúp xác định xem tên người dùng có trong nhóm không'''
    member_list = ['Nguyễn Xuân An', 'Võ Đình Đại', 'Nguyễn Tiến Đạt', 'Trần Ngọc Phúc', 'Nguyễn Hữu Thắng']
    if name in member_list:
        return True
    else:
        return False

