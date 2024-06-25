import pyautogui
import time

pyautogui.FAILSAFE = False

def move_mouse():
    # 获取屏幕宽度和高度
    screen_width, screen_height = pyautogui.size()
    
    
    # center point 
    ix, iy = screen_width/10, screen_height/10
    pyautogui.moveTo(ix, iy)
    
    # 计算每步的时间间隔
    total_steps = 10000
    interval = 60
    
    for i in range(0, total_steps, 1):
        
        # print(i)
        # 计算当前位置
        current_x, current_y = pyautogui.position()
        
        # 计算下一步的位置
        if i % 10 == 0:
            next_x, next_y = ix, iy
            print('reset mouse to center')
        else:
            next_x, next_y = current_x + 20, current_y + 20
            print('move to: ', next_x, next_y)
        
        # 移动鼠标到下一步的位置
        pyautogui.moveTo(next_x, next_y, duration=1)
        
        # 等待一段时间，模拟每分钟移动2公分
        time.sleep(interval)

if __name__ == "__main__":
    move_mouse()

