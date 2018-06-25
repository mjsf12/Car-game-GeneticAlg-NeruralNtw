import numpy as np
import pygame 
from ke import KerasGenetc
from itertools import chain

class Jogo():
    def __init__(self):
        self.altura = 600
        self.largura = 800
        self.nomeJanela = "Rede Neural + Algoritimo Genetico"
        self.funcionado = True
        self.rua=[]
        self.player = []
        self.inimigo = []
        self.individuos = 200
        self.Qinimigos = 25
        self.Vinimigo = 8
        self.Vindivi = 6
        self.largIni = []
        self.autoIni = []
        self.carregarTexturasObjetos()
        self.score = 0
        self.geracao = 1
        self.netw = KerasGenetc(self.individuos)
        self.mainClock = pygame.time.Clock()
        self.fps = 15      
        
    def carregarTexturasObjetos(self):
        for x in range(40):
            self.largIni.append(x*20)
        for x in range(12):
            self.autoIni.append(50*x)
        img = pygame.image.load('texturas/rua.jpg')        
        for y in range(2):
            rua = []
            rua.append(pygame.transform.scale(img, (800, 600)))
            rua.append(rua[0].get_rect())
            rua[1].topleft = (0,y*-600)
            self.rua.append(rua)
        img = pygame.image.load('texturas/player.png')
        for x in range(self.individuos):
            player = []
            player.append(pygame.transform.scale(img, (23, 47)))
            player.append(player[0].get_rect())
            player[1].topleft = (self.largura / 2, self.altura - 50)
            player.append(x)
            player.append([])
            self.player.append(player)

        img = [pygame.image.load('texturas/inimigo1.png'),pygame.image.load('texturas/inimigo2.png'),pygame.image.load('texturas/inimigo3.png')]
        for _ in range(self.Qinimigos):
            inimigo = []
            aux=np.random.choice(img)
            inimigo.append(pygame.transform.scale(aux, (23, 47)))
            inimigo.append(inimigo[0].get_rect())
            inimigo[1].topleft = (np.random.choice(self.largIni),-np.random.choice(self.autoIni))
            self.inimigo.append(inimigo)


    def executar(self):
        self.definirJanela()
        self.font = pygame.font.SysFont(None, 30)
        while (True):
            self.telaLoading()
            while(self.funcionado):
                self.loopGame()
                self.mainClock.tick(self.fps)
                pygame.display.update()
                self.score += 1
            self.telaLoading()
            self.score = 0
            #print(self.netw.scores())
            self.netw.refazer()
            self.refazer()
            self.funcionado = True
            self.geracao +=1
            #print(self.netw.scores())
    def refazer(self):
        for x in self.inimigo:
            x[1].topleft = (np.random.choice(self.largIni),-np.random.choice(self.autoIni))
        img = pygame.image.load('texturas/player.png')
        for x in range(self.individuos):
            player = []
            player.append(pygame.transform.scale(img, (23, 47)))
            player.append(player[0].get_rect())
            player[1].topleft = (self.largura / 2, self.altura - 50)
            player.append(x)
            player.append([])
            self.player.append(player)

    def loopGame(self):
        self.interpretarEventos()
        self.TestarColisao()
        self.desenharChao()
        self.desenharPlayars()
        self.desenharInimigos()
        self.desenharTexto()

    def telaLoading(self):
        self.escreverTexto("Carregando:",0,self.largura / 2,self.altura/2)
        pygame.display.update()

    
    def interpretarEventos(self):
        for Evento in pygame.event.get():
            if Evento.type == pygame.QUIT:
                self.funcionado = False
                
    def desenharChao(self):
        for rua in self.rua:
            rua[1].move_ip(0,2)
            self.janela.blit(rua[0], rua[1])
            if rua[1].topleft[1] >=600:
                rua[1].topleft =(0,-600)

    def desenharPlayars(self):
        for pl in self.player:
            aux = sorted(pl[3])
            aux = list(chain.from_iterable(aux))
            pl[3] = []
            aux2 = len(aux)
            if aux2 <= 20:
                for _ in range(int((20 -aux2)/2)):
                    aux.append(1)
                    aux.append(0)
            else:
                aux = aux[:20]
            self.netw.set_qual(pl[2])
            saida=self.netw.rodar(aux)[0]
            #print("individuo: " ,pl[2]," - ", aux)
            #print("Saida : " ,saida)
            fazer=np.argmax(saida)
            if fazer==0:
                pl[1].move_ip(0,-self.Vindivi)
            elif fazer==1:
                pl[1].move_ip(0,self.Vindivi)
            elif fazer==2:
                pl[1].move_ip(self.Vindivi,0)
            elif fazer==3:
                pl[1].move_ip(-self.Vindivi,0)
            self.janela.blit(pl[0],pl[1])
            
    def desenharInimigos(self):
        for ini in self.inimigo:
            ini[1].move_ip(0,self.Vinimigo)
            self.janela.blit(ini[0],ini[1])
            if ini[1].topleft[1]>= 600:
                ini[1].topleft = (np.random.choice(self.largIni), -30)
                
    def TestarColisao(self):
        for x in self.player:
            if x[1].right >= 800 or x[1].left <= 0 or x[1].top<=0 or x[1].bottom>=600:
                self.removerIndividuo(x)
                continue
            for y in self.inimigo:
                if x[1].colliderect(y[1]):
                    self.removerIndividuo(x)
                    break
                aux=self.verificarQuemPerto(x,y)
                if not aux == 0:
                    x[3].append([aux[0],aux[1]])
            aux = [0]
            aux.append(pygame.draw.rect(self.janela,(255,255,255),(0,0,0,0)))
            aux[1].center = (x[1].center[0],0)
            aux2=self.verificarQuemPerto(x,aux)
            if not aux2 == 0:
                x[3].append([aux2[0],aux2[1]])
            aux[1].center = (0,x[1].center[1])
            aux2 =self.verificarQuemPerto(x,aux)
            if not aux2 == 0:
                x[3].append([aux2[0],aux2[1]])
            aux[1].center = (x[1].center[0],600)
            aux2 =self.verificarQuemPerto(x,aux)
            if not aux2 == 0:
                x[3].append([aux2[0],aux2[1]])
            aux[1].center = (800,x[1].center[1])
            aux2 =self.verificarQuemPerto(x,aux)
            if not aux2 == 0:
                x[3].append([aux2[0],aux2[1]])
            #print(x[3])
            

        if len(self.player) == 0:
            self.funcionado = False
    
    def removerIndividuo(self,indi):
        self.netw.set_qual(indi[2])
        if self.score == 0:
            score = 1
        else:
            score = self.score
        self.netw.classificar(self.score)
        self.player.remove(indi)
    
    def verificarQuemPerto(self,player,inimigo):
        ply = np.array(player[1].center)
        ini = np.array(inimigo[1].center) 
        distAB = np.linalg.norm(ply-ini)
        if (distAB < 150):
            cosang = np.dot(ply, ini)
            sinang = (np.cross(ply, ini))
            angle = np.arctan2(sinang,cosang)
            if angle < 0:
                angle = angle + 1
            return ([distAB/150,angle])
        return 0

    def desenharTexto(self):
        self.escreverTexto("Pontos: ",self.score,0,0)
        self.escreverTexto("Indivious: ",len(self.player),0,20)
        self.escreverTexto("Gerações: ",self.geracao,0,40)
        self.escreverTexto("Fps: ",int(self.mainClock.get_fps()),0,60)

    def escreverTexto(self,texto, num,x,y):
        textobj = self.font.render(texto + str(num), 1, (255, 255, 255))
        textrect = textobj.get_rect()
        textrect.topleft = (x, y)
        self.janela.blit(textobj, textrect)  

    def definirJanela(self):
        pygame.init()
        self.janela=pygame.display.set_mode((self.largura, self.altura))
        pygame.display.set_caption(self.nomeJanela)
        pygame.mouse.set_visible(False)

if __name__ == "__main__":
    jan = Jogo()
    jan.executar()
