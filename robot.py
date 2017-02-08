import numpy as np

class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''
        self.run=0
        self.init = [0,0]

        # setup the initial location
        self.x=self.init[0] 
        self.y=self.init[1]
        
        # record the previous location, used by run-2
        self.prevX=0
        self.prevY=0

        # setup the goal of the maze
        self.goal_bounds = [maze_dim/2 - 1, maze_dim/2]

        # setup the heading of the maze
        self.heading = 'up'

        # setup the dimension of the maze
        self.maze_dim = maze_dim

        self.cost = 1

        # self.delta = [(0,1), # go up
        #              (1,0), # go right
        #              (0,-1), # go down
        #              (-1,0)] # go left
        # self.delta_name = ['^', '>', 'v', '<']

        self.delta = [(1,0), # go right
                     (0,1), # go up
                     (0,-1), # go down
                     (-1,0)] # go left
        # Remark: the order of delta is related to the order of direction the robot to navigate


        self.delta_name = ['>', '^', 'v', '<']

        # self.closed is matrix to record which positions of the maze have been visited
        self.closed=[[0 for col in range(maze_dim)] for row in range(maze_dim)]
        self.closed[self.init[0]][self.init[1]]=1
        # self.expand=[[-1 for i in range(maze_dim)] for j in range(maze_dim)] 
        self.action=[[-1]*maze_dim for i in range(maze_dim)]

        # Here I define a function called one_norm, which is used to generate a 
        # Heuristic Matrix for A-star search
        def one_norm(a,b):
            d=abs(a[0]-b[0])+abs(a[1]-b[1])
            return d

        # initialize the heuristic matrix
        self.heuristic=[[0 for col in range(maze_dim)] for row in range(maze_dim)]

        #The 1st heuristic matrix, H1
        for i in range(maze_dim):
            for j in range(maze_dim):
                self.heuristic[i][j]=min(one_norm((i,j),(maze_dim/2-1,maze_dim/2-1)),
                                   one_norm((i,j),(maze_dim/2-1,maze_dim/2)),
                                   one_norm((i,j),(maze_dim/2,maze_dim/2-1)),
                                   one_norm((i,j),(maze_dim/2,maze_dim/2)))

        

        # heuristic=[[30, 29, 28, 27, 26, 25, 24, 25, 24, 25, 24, 23],
        #  [25, 26, 27, 24, 23, 24, 23, 24, 23, 22, 21, 22],
        #  [24, 25, 28, 25, 22, 21, 22, 23, 18, 19, 20, 21],
        #  [23, 26, 27, 26, 19, 20, 21, 22, 17, 16, 15, 14],
        #  [22, 21, 20, 19, 18, 17, 16, 15, 14, 15, 14, 13],
        #  [19, 20, 19, 20, 19, 0, 0, 14, 13, 12, 13, 12],
        #  [18, 17, 18, 19, 20, 0, 0, 13, 12, 11, 12, 11],
        #  [15, 16, 17, 18, 5, 2, 1, 2, 11, 10, 9, 10],
        #  [14, 15, 6, 5, 4, 3, 4, 3, 4, 9, 8, 11],
        #  [13, 14, 15, 6, 5, 6, 5, 6, 5, 6, 7, 12],
        #  [12, 13, 14, 7, 10, 7, 8, 9, 6, 7, 8, 9],
        #  [11, 10, 9, 8, 9, 10, 11, 10, 7, 8, 9, 10]]

        # heuristic=[
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        # [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        # [2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,0],
        # [3,3,3,3,3,3,3,3,3,3,3,3,3,2,1,0],
        # [4,4,4,4,4,4,4,4,4,4,4,4,3,2,1,0],
        # [5,5,5,5,5,5,5,5,5,5,5,4,3,2,1,0],
        # [6,6,6,6,6,6,6,6,6,6,5,4,3,2,1,0],
        # [7,7,7,7,7,7,7,7,7,6,5,4,3,2,1,0],
        # [8,8,8,8,8,8,8,8,7,6,5,4,3,2,1,0],
        # [ 9, 9, 9, 9, 9, 9,9,8,7,6,5,4,3,2,1,0],
        # [10,10,10,10,10,10,9,8,7,6,5,4,3,2,1,0],
        # [11,11,11,11,11,10,9,8,7,6,5,4,3,2,1,0],
        # [12,12,12,12,11,10,9,8,7,6,5,4,3,2,1,0],
        # [13,13,13,12,11,10,9,8,7,6,5,4,3,2,1,0],
        # [14,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],
        # [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
        # ]

        # for i in range(maze_dim):
        #     for j in range(maze_dim):
        #         self.heuristic[i][j]=j

        # for i in range(maze_dim):
        #     for j in range(maze_dim):
        #         self.heuristic[i][j]=j


        # for i in range(maze_dim):
        #     for j in range(maze_dim):
        #         print(self.heuristic[i][j])
        #     print('\n')


        self.policy=[[' ']* self.maze_dim for i in range(self.maze_dim)]
        #print(self.heuristic)
        self.g=0 #g-value
        self.h=self.heuristic[self.x][self.y]
        self.f=self.g+self.h
        self.open=[[self.f,self.g,self.h,self.x,self.y]]
        self.path=[] #records the list of rotations

    
        self.count=0 # modify code **********

        #set two flag values
        self.found=False # True: search is complete
        self.resign=False # True: we caanot find expand

        ## For  backtracking
        self.needBacktrack=False
        self.bkMoveReady=False
        self.onBacktracking=False

        #self.backcounter=0



    def next_move(self, sensors):
        '''
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''

        ### The following definitions are designed for tuples ###
        #print(sensors)
        def equal(a,b):
            # check whether two tuples are equal: a, b are tuples
            if a[0]==b[0] and a[1]==b[1]:
                return True
            else:
                return False
        def add(a,b): # For adding two tuples
            # a, b, c are tuples
            c=(a[0]+b[0],a[1]+b[1])
            return c
        def tupleInList(a,L):
            # a is a tuple
            # L is a list of tuples
            r=False
            for i in range(len(L)):
                if equal(a,L[i]):
                    r=True
            return r
        def direction(delta):
            # from delta tuple (x,y) to direction string, such as:
            # 'up', 'down', 'right', 'left', etc.
            d='left'
            if delta==(0,-1):
                d='down'
            if delta==(1,0):
                d='right'
            if delta==(0,1):
                d='up'
            return d
        #########################################################

        if not self.found and not self.resign:
            if len(self.open) == 0:
                self.resign=True
            else:
                #self.open.sort()
                #self.open.reverse()

                next=self.open.pop()
                
                self.x=next[3]
                self.y=next[4]
                self.g=next[1]

                #self.expand[self.x][self.y]=self.count
                #self.count=self.count+1

        goal=[(self.maze_dim/2-1,self.maze_dim/2-1),(self.maze_dim/2-1,self.maze_dim/2),(self.maze_dim/2,self.maze_dim/2-1),(self.maze_dim/2,self.maze_dim/2)]
        
        def is_permissible(direct,heading,sensors):
            abs_dist={'left':0,'up':0,'right':0,'down':0} # left, up, right, down
            if heading=='up':
                abs_dist['left']=sensors[0]
                abs_dist['up']=sensors[1]
                abs_dist['right']=sensors[2]
            elif heading=='left':
                abs_dist['down']=sensors[0]
                abs_dist['left']=sensors[1]
                abs_dist['up']=sensors[2]
            elif heading=='right':
                abs_dist['up']=sensors[0]
                abs_dist['right']=sensors[1]
                abs_dist['down']=sensors[2]
            elif heading=='down':
                abs_dist['right']=sensors[0]
                abs_dist['down']=sensors[1]
                abs_dist['left']=sensors[2]
            if abs_dist[direct] >0:
                return True
            else:
                return False

        
        def isReasonableStep(x,y,delta):
            # check whether from (x,y) moves delta (e.g., (0,-1),(1,0),etc.) is a 
            # reasonable step
            result=False
            x2=x+delta[0]
            y2=y+delta[1]
            direct=direction(delta)
            if x2 >= 0 and x2 < self.maze_dim and y2 >= 0 and y2 < self.maze_dim:
                if self.closed[x2][y2]==0 and is_permissible(direct,self.heading,sensors):
                    result=True
            return result

        def determineHowToMove(delta,heading):
            if heading == 'left':
                if delta==(-1,0):
                    rotation=0
                    movement=1
                elif delta==(0,-1):
                    rotation=-90
                    movement=1
                elif delta==(0,1):
                    rotation=90
                    movement=1
            if heading == 'up':
                if delta==(0,1):
                    rotation=0
                    movement=1
                elif delta==(-1,0):
                    rotation=-90
                    movement=1
                elif delta==(1,0):
                    rotation=90
                    movement=1
            if heading == 'right':
                if delta==(1,0):
                    rotation=0
                    movement=1
                elif delta==(0,1):
                    rotation=-90
                    movement=1
                elif delta==(0,-1):
                    rotation=90
                    movement=1
            if heading == 'down':
                if delta==(0,-1):
                    rotation=0
                    movement=1
                elif delta==(1,0):
                    rotation=-90
                    movement=1
                elif delta==(-1,0):
                    rotation=90
                    movement=1
            return rotation, movement

        def rev_rotation(r):
            result=0
            if r==90:
                result=-90
            elif r==-90:
                result=90
            return result

        def updateHeading(rotation):
            if (rotation==90):
                if self.heading=='up':
                    self.heading='right'
                elif self.heading=='right':
                    self.heading='down'
                elif self.heading=='down':
                    self.heading='left'
                elif self.heading=='left':
                    self.heading='up'
            if (rotation==-90):
                if self.heading=='up':
                    self.heading='left'
                elif self.heading=='left':
                    self.heading='down'
                elif self.heading=='down':
                    self.heading='right'
                elif self.heading=='right':
                    self.heading='up'    
        def checkStep(policy,dir,x,y):
            # dir = '^','>','v','<'
            step=1
            if dir=='^':
                if policy[x][y]!=dir:
                    print('false check step!')
                elif policy[x][y+1]==dir:
                    step=2
                    if policy[x][y+2]==dir:
                        step=3
            elif dir=='v':
                if policy[x][y]!=dir:
                    print('false check step!')
                elif policy[x][y-1]==dir:
                    step=2
                    if policy[x][y-2]==dir:
                        step=3
            elif dir=='>':
                if policy[x][y]!=dir:
                    print('false check step!')
                elif policy[x+1][y]==dir:
                    step=2
                    if policy[x+2][y]==dir:
                        step=3
            elif dir=='<':
                if policy[x][y]!=dir:
                    print('false check step!')
                elif policy[x-1][y]==dir:
                    step=2
                    if policy[x-2][y]==dir:
                        step=3
            return step


        if self.run==0:
            if tupleInList((self.x,self.y),goal):
                self.found==True
                rotation='Reset'
                movement='Reset'
                goal=[(self.x,self.y)]

                a=goal[0][0]
                b=goal[0][1]

                self.policy[a][b]='*'
                while a != self.init[0] or b != self.init[1]:
                    a2=a-self.delta[self.action[a][b]][0]
                    b2=b-self.delta[self.action[a][b]][1]
                    self.policy[a2][b2]=self.delta_name[self.action[a][b]]
                    a=a2
                    b=b2

                # reset position and heading of the robot
                # for run-2
                x2=0
                y2=0
                self.heading='up'


                self.run=1
                
                
            else:
                if not self.onBacktracking:
                    fval_index_tuple=[] 
                    for i in range(len(self.delta)):
                        self.needBacktrack=True
                        if isReasonableStep(self.x,self.y,self.delta[i]):
                            x2=self.x+self.delta[i][0]
                            y2=self.y+self.delta[i][1]
                            g2=self.g+self.cost
                            h2=self.heuristic[x2][y2]
                            f2=g2+h2
                            fval_index_tuple.append((f2,g2,h2,x2,y2,i))

                    if fval_index_tuple: # check if fval_index_tuple is not an empty set
                        f2=min(fval_index_tuple)[0]
                        g2=min(fval_index_tuple)[1]
                        h2=min(fval_index_tuple)[2]
                        x2=min(fval_index_tuple)[3]
                        y2=min(fval_index_tuple)[4]
                        i=min(fval_index_tuple)[5]

                        self.open.append((f2,g2,h2,x2,y2))
                        self.closed[x2][y2]=1
                        self.action[x2][y2]=i
                        rotation, movement=determineHowToMove(self.delta[i],self.heading)
                        self.path.append(rotation)
                        self.needBacktrack=False
                    
                # needBacktrack means the robot get into dead-end, when
                # backtracking, there are two actions: move and rotate
                if self.needBacktrack==True:
                    if self.bkMoveReady==False: #move
                        movement=-1
                        rotation=0
                        if self.heading=='up':
                            x2=self.x
                            y2=self.y-1
                        elif self.heading=='down':
                            x2=self.x
                            y2=self.y+1
                        elif self.heading=='right':
                            x2=self.x-1
                            y2=self.y
                        elif self.heading=='left':
                            x2=self.x+1
                            y2=self.y

                        g2=self.g+self.cost
                        h2=self.heuristic[x2][y2]
                        f2=g2+h2
                        self.open.append([f2,g2,h2,x2,y2])
                        self.bkMoveReady=True
                        self.onBacktracking=True
                    else: #rotate
                        movement=0
                        r=self.path.pop()
                        rotation=rev_rotation(r)
                        x2=self.x
                        y2=self.y
                        g2=self.g+self.cost
                        h2=self.heuristic[x2][y2]
                        f2=g2+h2
                        self.open.append([f2,g2,h2,x2,y2])
                        self.needBacktrack=False
                        self.bkMoveReady=False
                        self.onBacktracking=False
        else: # self.run==1
            self.x=self.prevX
            self.y=self.prevY

            if self.policy[self.x][self.y]=='^':
                delta=(0,1)
                step=checkStep(self.policy,'^',self.x,self.y)
                self.prevY=self.y+step
            elif self.policy[self.x][self.y]=='>':
                delta=(1,0)
                step=checkStep(self.policy,'>',self.x,self.y)
                self.prevX=self.x+step
            elif self.policy[self.x][self.y]=='v':
                delta=(0,-1)
                step=checkStep(self.policy,'v',self.x,self.y)
                self.prevY=self.y-step
            elif self.policy[self.x][self.y]=='<':
                delta=(-1,0)
                step=checkStep(self.policy,'<',self.x,self.y)
                self.prevX=self.x-step
            elif self.policy[self.x][self.y]=='*':
                print("goal!")

            rotation, movement=determineHowToMove(delta,self.heading)
            movement=step
            

            
        # update the heading of the robot
        updateHeading(rotation)

        
        #print(self.heading)
        #print 'Rotation: {}, Movement: {}, Location: {}, Heading: {}'.format(rotation, movement, (self.x, self.y), self.heading)
        print '[{}, \'{}\'],'.format((self.x, self.y), self.heading)
        return rotation, movement