import pygame
import sys
from collections import deque
import heapq
def bfs(maze, screen):
    start = Node(state=maze.start, parent=None, action=None)
    frontier = QueueFrontier()
    frontier.add(start)
    explored = set()
    steps = []  
    while True:
        if frontier.empty():
            return None  
        node = frontier.remove()
        steps.append(node.state)  

        draw_maze(screen, maze, show_solution=True, current=node.state)
        pygame.display.flip()  
        pygame.time.wait(50)  

        if node.state == maze.goal:
            
            solution_path = backtrack_solution(node) 
            draw_maze(screen, maze, show_solution=True, solution_path=solution_path)
            pygame.display.flip()  
            return solution_path  

        explored.add(node.state)

        for action, state in maze.neighbors(node.state):
            if not frontier.contains_state(state) and state not in explored:
                child = Node(state=state, parent=node, action=action)
                frontier.add(child)

def dfs(maze, screen):
    start = Node(state=maze.start, parent=None, action=None)
    stack = [start] 

    explored = set()

    while stack:
        node = stack.pop()  
        
        draw_maze(screen, maze, show_solution=True, current=node.state)
        pygame.display.flip()  
        pygame.time.wait(50)  

        if node.state == maze.goal:
            
            draw_maze(screen, maze, show_solution=True, solution_path=backtrack_solution(node))
            pygame.display.flip()
            return backtrack_solution(node)

        explored.add(node.state)

        for action, state in maze.neighbors(node.state):
            if state not in explored and not any(n.state == state for n in stack):
                child = Node(state=state, parent=node, action=action)
                stack.append(child)

def greedy_bfs(maze, screen):
    start = Node(state=maze.start, parent=None, action=None)
    priority_queue = [] 
    heapq.heappush(priority_queue, (0, start))  

    explored = set()

    while priority_queue:
        
        _, node = heapq.heappop(priority_queue)

        
        draw_maze(screen, maze, show_solution=True, current=node.state)
        pygame.display.flip()  
        pygame.time.wait(50) 

        if node.state == maze.goal:
            
            draw_maze(screen, maze, show_solution=True, solution_path=backtrack_solution(node))
            pygame.display.flip()
            return backtrack_solution(node)

        explored.add(node.state)

        
        for action, state in maze.neighbors(node.state):
            if state not in explored and not any(n[1].state == state for n in priority_queue):
                
                h = heuristic(state, maze.goal)
                child = Node(state=state, parent=node, action=action)
                heapq.heappush(priority_queue, (h, child))

def a_star(maze, screen):
    start = Node(state=maze.start, parent=None, action=None, cost=0)
    frontier = []
    total_cost = heuristic(start.state, maze.goal)
    heapq.heappush(frontier, (total_cost, start))  
    explored = set()
    costs = {start.state: 0}

    while frontier:
        _, node = heapq.heappop(frontier)

    
        draw_maze(screen, maze, show_solution=True, current=node.state)
        pygame.display.flip() 
        pygame.time.wait(50)  

        if node.state == maze.goal:
            solution_path = backtrack_solution(node)
            return solution_path

        explored.add(node.state)

        for action, state in maze.neighbors(node.state):
            new_cost = costs[node.state] + 1  
            if state not in costs or new_cost < costs[state]:
                costs[state] = new_cost
                total_cost = new_cost + heuristic(state, maze.goal)
                child = Node(state=state, parent=node, action=action, cost=total_cost)
                heapq.heappush(frontier, (total_cost, child))

    return None
def heuristic(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])



def backtrack_solution(node):
    actions = []
    cells = []
    while node.parent is not None:
        actions.append(node.action)
        cells.append(node.state)
        node = node.parent
    actions.reverse()
    cells.reverse()
    return cells  

class Node():
    def __init__(self, state, parent, action,cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
    def __lt__(self, other):
        return self.cost < other.cost
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
class QueueFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node

class Maze():
    def __init__(self, filename):
        with open(filename) as f:
            contents = f.read()

        if contents.count("A") != 1 or contents.count("B") != 1:
            raise Exception("maze must have exactly one start point and one goal")

        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                if contents[i][j] == "A":
                    self.start = (i, j)
                    row.append(False)
                elif contents[i][j] == "B":
                    self.goal = (i, j)
                    row.append(False)
                elif contents[i][j] == " ":
                    row.append(False)
                else:
                    row.append(True)
            self.walls.append(row)

        self.solution = None

    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def solve(self):
        self.num_explored = 0
        start = Node(state=self.start, parent=None, action=None)
        frontier = QueueFrontier()
        frontier.add(start)

        self.explored = set()

        while True:
            if frontier.empty():
                raise Exception("no solution")

            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = cells  
                return

            self.explored.add(node.state)

            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    frontier.add(child)

# Función para dibujar el laberinto

rustic_texture = pygame.image.load("punto1.png")
rustic_texture = pygame.transform.scale(rustic_texture, (30, 30))  # Ajusta el tamaño según el tamaño de la celda

# Función para dibujar el laberinto
def draw_maze(screen, maze, show_solution=False, current=None, solution_path=None, path_color=(220, 235, 113)):
    cell_size = 30  # Ajusta el tamaño de cada celda
    for i in range(maze.height):
        for j in range(maze.width):
            if maze.walls[i][j]:
                # Dibuja la textura de la pared
                screen.blit(rustic_texture, (j * cell_size, i * cell_size))
            elif (i, j) == maze.start:
                pygame.draw.rect(screen, (255, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
            elif (i, j) == maze.goal:
                pygame.draw.rect(screen, (0, 171, 28), (j * cell_size, i * cell_size, cell_size, cell_size))
            if show_solution and solution_path and (i, j) in solution_path:
                pygame.draw.rect(screen, path_color, (j * cell_size, i * cell_size, cell_size, cell_size))
            if current and (i, j) == current:
                
                pygame.draw.rect(screen, (0, 0, 255), (j * cell_size, i * cell_size, cell_size, cell_size))

def game_loop(maze):
    pygame.init()
    screen = pygame.display.set_mode((maze.width * 30, maze.height * 30 + 50)) 
    clock = pygame.time.Clock()

    current_position = maze.start
    start_time = pygame.time.get_ticks()  

    while True:
        screen.fill((237, 240, 252))
        draw_maze(screen, maze)
        pygame.draw.rect(screen, (255, 0, 0), (current_position[1] * 30, current_position[0] * 30, 30, 30))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and not maze.walls[current_position[0] - 1][current_position[1]]:
                    current_position = (current_position[0] - 1, current_position[1])
                elif event.key == pygame.K_DOWN and not maze.walls[current_position[0] + 1][current_position[1]]:
                    current_position = (current_position[0] + 1, current_position[1])
                elif event.key == pygame.K_LEFT and not maze.walls[current_position[0]][current_position[1] - 1]:
                    current_position = (current_position[0], current_position[1] - 1)
                elif event.key == pygame.K_RIGHT and not maze.walls[current_position[0]][current_position[1] + 1]:
                    current_position = (current_position[0], current_position[1] + 1)

       
        if current_position == maze.goal:
            end_time = pygame.time.get_ticks() 
            time_taken = (end_time - start_time) // 1000  
          
            font = pygame.font.Font(None, 48)
            message = f"¡Felicidades! Tardaste {time_taken} segundos."
            text = font.render(message, True, (0, 0, 0))
            screen.fill((237, 240, 252))
            screen.blit(text, (50, maze.height * 30 // 2))
            pygame.display.flip()
            pygame.time.wait(30000)  
            return  

        pygame.display.flip()
        clock.tick(60000)


def show_solution(maze):
    screen = pygame.display.set_mode((maze.width * 30, maze.height * 30 + 50))  # Ajusta el tamaño de la ventana
    clock = pygame.time.Clock()

   
    algorithms = ["BFS", "DFS", "Greedy", "A*"]
    selected_algorithm = None
    buttons = [
        {"text": algo, "pos": (50, 50 + i * 60), "visible": True} for i, algo in enumerate(algorithms)]

   
    background_image = pygame.image.load("fondo2.png").convert()  
    background_image = pygame.transform.scale(background_image, (maze.width * 30, maze.height * 30 + 50))

    while selected_algorithm is None:
        screen.blit(background_image, (0, 0))  
        
        draw_maze(screen, maze)  

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                for button in buttons:
                    if button["pos"][0] <= mouse_x <= button["pos"][0] + 200 and button["pos"][1] <= mouse_y <= button["pos"][1] + 40:
                        selected_algorithm = button["text"]  
                        button["visible"] = False  

        
        for button in buttons:
            if button["visible"]:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                is_hovered = button["pos"][0] <= mouse_x <= button["pos"][0] + 200 and button["pos"][1] <= mouse_y <= button["pos"][1] + 40
                draw_button(screen, button["text"], button["pos"], (200, 40), font, (0, 150, 0), (0, 100, 0), is_hovered)

        pygame.display.flip()
        clock.tick(60)

   
    if selected_algorithm == "BFS":
        bfs(maze, screen)
        pygame.time.wait(5000)
    elif selected_algorithm == "DFS":
        dfs(maze, screen)
        pygame.time.wait(5000)
    elif selected_algorithm == "Greedy":
        greedy_bfs(maze, screen)
        pygame.time.wait(5000)
    elif selected_algorithm == "A*":
        solution_path = a_star(maze, screen)
    if solution_path:
        draw_maze(screen, maze, show_solution=True, solution_path=solution_path)
        pygame.display.flip()
        pygame.time.wait(5000) 
    pass

def draw_button(screen, text, pos, size, font, hover_color, default_color, is_hovered):
    color = hover_color if is_hovered else default_color
    pygame.draw.rect(screen, color, (pos[0], pos[1], size[0], size[1]), border_radius=10)  

    
    pygame.draw.rect(screen, (0, 0, 0), (pos[0], pos[1], size[0], size[1]), 2, border_radius=10)

    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(pos[0] + size[0] // 2, pos[1] + size[1] // 2))
    screen.blit(text_surface, text_rect)


def level_selection_menu():
    pygame.init()
    screen = pygame.display.set_mode((300, 200))
    font = pygame.font.Font(None, 36)

   
    background_image = pygame.image.load("una.png").convert()  
    background_image = pygame.transform.scale(background_image, (300, 200))  
    levels = ["Fácil", "Intermedio", "Difícil"]
    selected_level = None

    while selected_level is None:
        screen.blit(background_image, (0, 0))  

        button_positions = [(50, 50 + i * 60) for i in range(len(levels))]
        button_size = (200, 30)

        for i, level in enumerate(levels):
            mouse_x, mouse_y = pygame.mouse.get_pos()
            is_hovered = button_positions[i][0] <= mouse_x <= button_positions[i][0] + button_size[0] and button_positions[i][1] <= mouse_y <= button_positions[i][1] + button_size[1]
            draw_button(screen, level, button_positions[i], button_size, font, (0, 150, 0), (0, 200, 0), is_hovered)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                for i, level in enumerate(levels):
                    if button_positions[i][0] <= mouse_x <= button_positions[i][0] + button_size[0] and button_positions[i][1] <= mouse_y <= button_positions[i][1] + button_size[1]:
                        selected_level = f"laberinto{i+2}.txt"

        pygame.display.flip()

    return selected_level

if __name__ == "__main__":
    selected_level = level_selection_menu()
    maze = Maze(selected_level)

   
    pygame.init()
    screen = pygame.display.set_mode((maze.width * 30, maze.height * 30 + 50))  
    font = pygame.font.Font(None, 36)
    background_image_menu = pygame.image.load("fondo2.png").convert()  
    background_image_menu = pygame.transform.scale(background_image_menu, (maze.width * 30, maze.height * 30 + 50))

    buttons = [
        {"text": "Jugar", "pos": (200, maze.height * 30)},
        {"text": "Ver Solución", "pos": (400, maze.height * 30)},
    ]

    while True:
        screen.blit(background_image_menu, (0, 0))  
        
      
        draw_maze(screen, maze)  

        for button in buttons:
            is_hovered = button["pos"][0] <= pygame.mouse.get_pos()[0] <= button["pos"][0] + 100 and button["pos"][1] <= pygame.mouse.get_pos()[1] <= button["pos"][1] + 40
            draw_button(screen, button["text"], button["pos"], (100, 40), font, (0, 150, 0), (0, 100, 0), is_hovered)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                for button in buttons:
                    if button["pos"][0] <= mouse_x <= button["pos"][0] + 100 and button["pos"][1] <= mouse_y <= button["pos"][1] + 40:
                        if button["text"] == "Jugar":
                            game_loop(maze)
                        elif button["text"] == "Ver Solución":
                            show_solution(maze)

        pygame.display.flip()
        