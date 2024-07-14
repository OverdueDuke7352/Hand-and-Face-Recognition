#on python 3.9.13 is worked!
import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

width, height = 1000, 700

pygame.init()  # 初始化Pygame
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Bubble Shooter")
background = pygame.image.load("Games/Res/BackgroundBlue.jpg").convert_alpha()
background = pygame.transform.scale(background, (width, height))
clock = pygame.time.Clock()

class IntroBubble(pygame.sprite.Sprite):
    def __init__(self, Bubble, x, y, speed):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = pygame.image.load(Bubble).convert_alpha()
        self.image = pygame.transform.scale(self.image, (75, 75))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = speed
        self.radius = 25

    def update(self):
        screen.blit(self.image, self.rect)
        self.rect.y -= self.speed
        if self.rect.y <= 0:
            self.speed  = 0
            self.rect.y = 0

def windows(background):
    font = pygame.font.Font(None, 36)
    running = True
    next_bubble_batch = pygame.USEREVENT + 1
    pygame.time.set_timer(next_bubble_batch, 1000)

    all_bubbles = pygame.sprite.Group()
    IntroBubble.containers = all_bubbles

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running = False
            elif event.type == next_bubble_batch:
                for _ in range(random.randint(2, 3)):
                    IntroBubble("Games/Res/Bubble.png", random.randint(0, width - 75), height, random.randint(1, 2))

        screen.blit(background, (0, 0))
        start_text = font.render('Enter ANY KEY to Start', True, (255, 255, 255))
        screen.blit(start_text, (width // 2 - start_text.get_width() // 2, height // 2 - start_text.get_height() // 2))

        all_bubbles.update()
        pygame.display.flip()
        clock.tick(60)

class MainWinBubble(pygame.sprite.Sprite):
    def __init__(self, Bubble, x, y, speed):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = pygame.image.load(Bubble).convert_alpha()
        self.image = pygame.transform.scale(self.image, (75, 75))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = speed

    def update(self, score, position):
        self.rect.y -= self.speed
        if self.rect.y <= 0:
            self.kill()
        else:
            if self.rect.collidepoint(position):
                pop = pygame.image.load("Games/Res/Pop.png").convert_alpha()
                pop = pygame.transform.scale(pop, (135, 120))
                self.image = pop
                screen.blit(pop, (self.rect.x, self.rect.y))
                score += 10
                self.kill()
        screen.blit(self.image, self.rect)
        return score

    def collide(self, all_bubble_list):
        collections = pygame.sprite.spritecollide(self, all_bubble_list, False, pygame.sprite.collide_circle)
        for each in collections:
            if each.speed == 0:
                self.speed = 0

def get_frame(cap):
    ret, frame = cap.read() #讀取鏡頭畫面
    frame = cv2.resize(frame, (width, height)) #調整畫面大小降低延遲
    if not ret: #如果沒有讀取到畫面
        return None, (0, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #轉換畫面顏色
    results = hands.process(frame) #處理手部定位點
    x1, y1 = 0, 0 #初始化手部座標
    if results.multi_hand_landmarks: #如果偵測到手部
        hand_landmarks = results.multi_hand_landmarks[0].landmark #取得手部定位點
        single_hand = hand_landmarks[8] #取得食指指尖定位點
        x1, y1 = width - int(single_hand.x * width), int(single_hand.y * height) #取得食指指尖座標
        for hand_landmarks in results.multi_hand_landmarks: #遍歷每一隻手
            mp_drawing.draw_landmarks( #繪製手部定位點
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    frame = np.rot90(frame) #旋轉畫面
    frame = pygame.surfarray.make_surface(frame) #轉換畫面格式
    return frame, (x1, y1) #返回畫面和手部座標

def end_screen(total_time):
    font = pygame.font.Font(None, 48)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))
        end_text = font.render(f"Game Over! Total Time: {total_time:.2f} seconds", True, (0, 0, 0))
        screen.blit(end_text, (width // 2 - end_text.get_width() // 2, height // 2 - end_text.get_height() // 2 - 50))
        pygame.display.flip()
        clock.tick(60)

def main_window(background):
    font = pygame.font.Font(None, 36)
    bubble_drop = pygame.USEREVENT + 2
    pygame.time.set_timer(bubble_drop, 2000)
    high_score = 500
    score = 0
    game_over = False
    start_time = time.time()
    all_bubble_list = pygame.sprite.Group()
    all_sprites = pygame.sprite.Group()
    MainWinBubble.containers = all_sprites, all_bubble_list
    windows(background)
    cap = cv2.VideoCapture(2)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == bubble_drop:
                for _ in range(random.randint(1, 3)):
                    MainWinBubble("Games/Res/Bubble.png", random.randint(0, width - 75), height, random.randint(1, 4))

        frame, position = get_frame(cap)
        screen.blit(frame, (0, 0))
        score_text = font.render(f"Scores: {score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        if not game_over:
            for bubble in all_bubble_list:
                score = bubble.update(score, position)
            for bubble in all_bubble_list:
                all_bubble_list.remove(bubble)
                bubble.collide(all_bubble_list)
                all_bubble_list.add(bubble)
            if score >= high_score:
                game_over = True
        else:
            all_sprites.clear(screen, background)
            all_sprites.update(score)

        pygame.display.flip()
        clock.tick(60)

    total_time = time.time() - start_time
    cap.release()
    cv2.destroyAllWindows()
    end_screen(total_time)
    pygame.quit()

if __name__ == "__main__":
    main_window(background)
