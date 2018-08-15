from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import bitarray
import struct
import threading
from keras.models import model_from_yaml
import tensorflow as tf
graph = tf.get_default_graph()
class KerasGenetc():
    def __init__(self,n):
        self.networks = []
        self.tam = n
        self.qual = 0
        theads = []
        for x in range(8):
            theads.append(Theads_Criar("thead - " + str(x),self.tam/8))
        for x in theads:
            x.start()
        for x in theads:
            x.join()
            self.networks = self.networks +x.net
            #self.networks.append(self.create_neural())
        print (len(self.networks))
        exit()
    def create_neural(self,Peso=0):
        with graph.as_default():
            model = Sequential()
            model.add(Dense(6,  input_dim=20 ,bias_initializer='random_uniform'))
            model.add(Dense(12,bias_initializer='random_uniform'))
            model.add(Dense(5, activation='softmax' ,bias_initializer='random_uniform'))
            if Peso == 0:
                Peso=model.get_weights()
            else:
                model.set_weights(Peso)
            gene = self.toGenes(Peso)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return ([0,model,gene])
    
    def toGenes(self,a):
        aux = []
        for x in range(6):
            if(not x % 2 == 0):
                for y in a[x]:
                    aux.append(y)           
            else:
                for y in a[x]:
                    for z in y:
                        aux.append(z)
        return aux

    def GenToArray(self,gen=0):
        if gen == 0:
            aux=self.networks[self.qual][2]
        else:
            aux = gen
        aux2=[]
        for _ in range(1045):
            saida = bitarray.bitarray()
            for _  in range(32):
                saida.append(aux.pop())
            saida = saida.tobytes()
            saida = struct.unpack('f',saida)
            aux2.append(saida[0] % 1)
        return aux2;
    
    def ArrayToPesos(self,array):
        fin=[]
        pri= []
        for y in range(20):
            aux =[]
            for x in range(6):
                aux.append(array[y*6+x])
            pri.append(aux)
        sec =[]       
        for x in range(6):
            sec.append(array[20*6+x])
        tri = []
        for x in range(6):
            aux=[]
            for y in range(12):
                aux.append(array[21*6+x*12+y])
            tri.append(aux)
        fort = []
        for x in range(12):
            fort.append(array[21*6+6*12+x])
        fiv = []
        for x in range(12):
            aux=[]
            for y in range(5):
                aux.append(array[21*6+7*12+x*5+y])
            fiv.append(aux)
        sex = []
        for x in range(5):
            sex.append(array[21*6+7*12+5*12+x])
        fin.append(np.array(pri,dtype="float32"))
        fin.append(np.array(sec,dtype="float32"))
        fin.append(np.array(tri,dtype="float32"))
        fin.append(np.array(fort,dtype="float32"))
        fin.append(np.array(fiv,dtype="float32"))
        fin.append(np.array(sex,dtype="float32"))
        return fin
    
    
    def set_qual(self,n):
        self.qual = n

    def rodar(self,array):
        return self.networks[self.qual][1].predict(np.array([array]))
    
    def classificar(self,score):
        self.networks[self.qual][0]=score
        
    def scores(self):
        volta=[]
        for net in self.networks:
            volta.append(net[0])
        return volta

    def getKey(self,item):
        return item[0]

    def GerarGenes(self):
        aux = []
        self.ClassificaR()
        for _ in range(self.tam-50):
            rand=np.random.uniform(0,self.networks[-1][0])
            pai=self.Seleci(rand)
            mae = pai
            while pai[0] == mae[0]:
                rand=np.random.uniform(0,self.networks[-1][0])
                mae = self.Seleci(rand)
            Gfilho =self.reprodu(pai,mae)
            # ("mae: ",mae[2])
            #print ("pai : ",pai[2])
            #print ("Filho:",Gfilho)
            #print ("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            aux.append(Gfilho)
        return aux
        
    def refazer(self):
        self.networks.sort(key=self.maxs)
        self.networks=self.networks[-10:]
        GenesFilhos= self.GerarGenes()
        for x in self.networks:
            print (x[0])
        for x in GenesFilhos:
            aux=self.ArrayToPesos(x)
            rede=self.create_neural(aux)
            self.networks.append(rede)
            
        for _ in range(40):
            rede=self.create_neural()
            self.networks.append(rede)
            
        for x in self.networks:
            print (x[0])       

            
    def reprodu(self,pai,mae):
        Gpai=pai[2]
        Gmae=mae[2]
        rand=np.random.random_integers(0,len(Gmae)-1)
        Gfilho= Gmae[:rand] + Gpai[rand:] 
        if np.random.uniform(0,100) >= 90:
            rand=np.random.random_integers(0,len(Gfilho)-1)
            rand2=np.random.uniform(-1,1)
            Gfilho[rand] = rand2
        return Gfilho

    def Seleci(self,num):
        for i in self.networks:
            if num <= i[0]:
                return i
    
    def ClassificaR(self):
        maxN=0
        for i in self.networks:
            maxN += i[0]
            i[0] =maxN
        
    def getpesos(self,model):
        return model.get_weights()

    def getpesosP(self):
        #print (self.networks[self.qual][1].get_weights())
        return self.toGenes(self.networks[self.qual][1].get_weights())
    
    def maxs(self,x):
        return x[0]
    
    def SalvarRedeMelhor(self,x):
        aux = max(self.networks, key=self.maxs)
        model = aux[1]
        model_yaml = model.to_yaml()
        with open("Geracao "+str(x)+" Ganhador.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights("Geracao "+str(x)+" Ganhador.h5")

class Theads_Criar(threading.Thread):

    def __init__(self,nome,qtd,peso=0):
        threading.Thread.__init__(self)
        self.name = nome
        self.qtd = int(qtd)
        self.peso = peso
        self.net = []
        #print(qtd)

    def run(self):
        for x in range(self.qtd):
            print(self.name + " - fazendo rede:"+ str(x))
            self.net.append(KerasGenetc.create_neural(self,self.peso))
    def getpesos(self,model):
        return model.get_weights()    

    def toGenes(self,a):
        aux = []
        for x in range(6):
            if(not x % 2 == 0):
                for y in a[x]:
                    aux.append(y)           
            else:
                for y in a[x]:
                    for z in y:
                        aux.append(z)
        return aux