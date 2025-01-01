#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robï¿½tica Computacional 
# Grado en Ingenierï¿½a Informï¿½tica (Cuarto)
# Prï¿½ctica 5:
#     Simulaciï¿½n de robots mï¿½viles holonï¿½micos y no holonï¿½micos.

#localizacion.py

import sys
from math import *
from robot import robot
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# ******************************************************************************
# Declaraciï¿½n de funciones

def distancia(a,b):
  # Distancia entre dos puntos (admite poses)
  return np.linalg.norm(np.subtract(a[:2],b[:2]))

def angulo_rel(pose,p):
  # Diferencia angular entre una pose y un punto objetivo 'p'
  w = atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]
  while w >  pi: w -= 2*pi
  while w < -pi: w += 2*pi
  return w

def mostrar(objetivos,ideal,trayectoria):
  # Mostrar objetivos y trayectoria:
  #plt.ion() # modo interactivo
  # Fijar los bordes del grï¿½fico
  objT   = np.array(objetivos).T.tolist()
  trayT  = np.array(trayectoria).T.tolist()
  ideT   = np.array(ideal).T.tolist()
  bordes = [min(trayT[0]+objT[0]+ideT[0]),max(trayT[0]+objT[0]+ideT[0]),
            min(trayT[1]+objT[1]+ideT[1]),max(trayT[1]+objT[1]+ideT[1])]
  centro = [(bordes[0]+bordes[1])/2.,(bordes[2]+bordes[3])/2.]
  radio  = max(bordes[1]-bordes[0],bordes[3]-bordes[2])*.75
  plt.xlim(centro[0]-radio,centro[0]+radio)
  plt.ylim(centro[1]-radio,centro[1]+radio)
  # Representar objetivos y trayectoria
  idealT = np.array(ideal).T.tolist()
  plt.plot(idealT[0],idealT[1],'-g')
  plt.plot(trayectoria[0][0],trayectoria[0][1],'or')
  r = radio * .1
  for p in trayectoria:
    plt.plot([p[0],p[0]+r*cos(p[2])],[p[1],p[1]+r*sin(p[2])],'-r')
    #plt.plot(p[0],p[1],'or')
  objT   = np.array(objetivos).T.tolist()
  plt.plot(objT[0],objT[1],'-.o')
  plt.show()
  input()
  plt.clf()

# Centro = vector [x, y]
def localizacion(balizas, real, ideal, centro, radio, limite_incremento, incremento_angulo, mostrar=0):
  # Buscar la localizaciï¿½n mï¿½s probable del robot, a partir de su sistema
  # sensorial, dentro de una regiï¿½n cuadrada de centro "centro" y lado "2*radio".
  mejor_pose = ideal.pose()
  error_mejor_pose = 1000 # Valor arbitrario alto
  
  # Bï¿½squeda en "pirï¿½mide", cada iteraciï¿½n elige una cuadrï¿½cula
  incremento = radio
  while incremento > limite_incremento:
    imagen = []
    print("Radio: ", radio, "Incremento: ", incremento)
    # Buscar mejores coordenadas
    for y in np.arange(-radio, radio, incremento):
      imagen.append([])
      for x in np.arange(-radio, radio, incremento):
        ideal.set(centro[0] + x + radio/2, centro[1] + y + radio/2, ideal.orientation)
        error_actual = real.measurement_prob(ideal.sense(balizas), balizas)
        imagen[-1].append(error_actual)
        
        if error_actual < error_mejor_pose:
          error_mejor_pose = error_actual
          mejor_pose = ideal.pose()
    
    if mostrar:
      #plt.ion() # modo interactivo
      plt.xlim(centro[0]-radio,centro[0]+radio)
      plt.ylim(centro[1]-radio,centro[1]+radio)
      imagen.reverse()
      plt.imshow(imagen,extent=[centro[0]-radio,centro[0]+radio,\
                                centro[1]-radio,centro[1]+radio])
      balT = np.array(balizas).T.tolist();
      plt.plot(balT[0],balT[1],'or',ms=10)
      plt.plot(mejor_pose[0],mejor_pose[1],'D',c='#ff00ff',ms=10,mew=2)
      plt.plot(real.x, real.y, 'D',c='#00ff00',ms=10,mew=2)
      plt.show()
      # input()
      plt.clf()
    
    # Actualizamos la "pirï¿½mide" 
    centro = [mejor_pose[0], mejor_pose[1]]
    radio /= 2
    incremento = radio
      	
  # Buscar mejor orientaciï¿½n
  mejor_orientacion = ideal.orientation
  for angulo in np.arange(-pi, pi, incremento_angulo):
    ideal.set(mejor_pose[0], mejor_pose[1], angulo)
    error_actual = real.measurement_prob(ideal.sense(balizas), balizas)

    if error_actual < error_mejor_pose:
      error_mejor_pose = error_actual
      mejor_pose = ideal.pose()
  
  print("Error mejor pose: ", error_mejor_pose)
  return mejor_pose

# ******************************************************************************

# Parametros
UMBRAl_DIF_TRAYECTORIA = 0.1 # Diferencia entre mediciones ideales y reales a partir de la que se corrige la pose ideal
LIMITE_INCREMENTO = 0.05 # Precisiï¿½n al estimar la localizaciï¿½n
INCREMENTO_ANGULO = 0.01 # incremento al probar la mejor orientacion
MARGEN = 1 # Borde que se aï¿½ade en la localizaciï¿½n inical sobre el area de las balizas
# RADIO = 1 # Constante arbitraria

# Definiciï¿½n del robot:
P_INICIAL = [0.,4.,0.] # Pose inicial (posicion y orientacion)
V_LINEAL  = .7         # Velocidad lineal    (m/s)
V_ANGULAR = 40.       # Velocidad angular   (º/s)
FPS       = 10.        # Resolucion temporal (fps)

HOLONOMICO = 1
GIROPARADO = 0
LONGITUD   = .2

# Definiciï¿½n de trayectorias:
trayectorias = [
    [[1,3]],
    [[0,2],[4,2]],
    [[2,4],[4,0],[0,0]],
    [[2,4],[2,0],[0,2],[4,2]],
    [[2+2*sin(.8*pi*i),2+2*cos(.8*pi*i)] for i in range(5)]
    ]

# Definiciï¿½n de los puntos objetivo:
if len(sys.argv)<2 or int(sys.argv[1])<0 or int(sys.argv[1])>=len(trayectorias):
  sys.exit(sys.argv[0]+" <indice entre 0 y "+str(len(trayectorias)-1)+">")
objetivos = trayectorias[int(sys.argv[1])]

# Definiciï¿½n de constantes:
EPSILON = .01                # Umbral de distancia
V = V_LINEAL/FPS            # Metros por fotograma
W = V_ANGULAR*pi/(180*FPS)  # Radianes por fotograma

ideal = robot()
ideal.set_noise(0,0,.1)   # Ruido lineal / radial / de sensado
ideal.set(*P_INICIAL)     # operador 'splat'

real = robot()
real.set_noise(.01,.01,.1)  # Ruido lineal / radial / de sensado
real.set(*P_INICIAL)

random.seed(0)
tray_ideal = [ideal.pose()]  # Trayectoria percibida
tray_real = [real.pose()]     # Trayectoria seguida

tiempo  = 0.
espacio = 0.
#random.seed(0)
# OJO MODIFICADO python > 3.9
random.seed(int(datetime.now().timestamp()))

######################
# Exploracion inicial#
######################
# Se toma un radio segï¿½n las coordenadas mï¿½ximas y mï¿½nimas de balizas + margen

# Obtener las componentes por separado
valores_x, valores_y = zip(*objetivos)

# Calcular los mï¿½nimos y mï¿½ximos
min_x, max_x = min(valores_x), max(valores_x)
min_y, max_y = min(valores_y), max(valores_y)

centro_inicial = [(min_x + max_x) / 2, (min_y + max_y) / 2]
# Calculamos el radio como el mï¿½ximo entre la distancia vertical u horizontal
radio_inicial = max(abs(max_x - min_x), abs(max_y - min_y))/2 + MARGEN
# radio_inicial = sqrt((max_x - min_x)**2 + (max_y - min_y)**2) / 2 + MARGEN

inicial = localizacion(objetivos, real, ideal, centro_inicial, radio_inicial, LIMITE_INCREMENTO, INCREMENTO_ANGULO, 1)
ideal.set(*inicial)
  
for punto in objetivos:
  while distancia(tray_ideal[-1],punto) > EPSILON and len(tray_ideal) <= 1000:
    pose = ideal.pose()

    w = angulo_rel(pose,punto)
    if w > W:  w =  W
    if w < -W: w = -W
    v = distancia(pose,punto)
    if (v > V): v = V
    if (v < 0): v = 0

    if HOLONOMICO:
      if GIROPARADO and abs(w) > .01:
        v = 0
      ideal.move(w,v)
      real.move(w,v)
    else:
      ideal.move_triciclo(w,v,LONGITUD)
      real.move_triciclo(w,v,LONGITUD)
    tray_ideal.append(ideal.pose())
    tray_real.append(real.pose())
    
    # Comparar baliza real e ideal; si la diferencia es grande, localizar
    diferencia_mediciones = real.measurement_prob(ideal.sense(objetivos), objetivos)
    print("Medicion: ", diferencia_mediciones)
    # Si la diferencia es grande, localizar
    if (diferencia_mediciones > UMBRAl_DIF_TRAYECTORIA):
      # radio de localizaciï¿½n = 2 * diferencia
      ideal.set(*localizacion(objetivos, real, ideal, [ideal.x, ideal.y], 2*diferencia_mediciones, LIMITE_INCREMENTO, INCREMENTO_ANGULO)) # actualizar pose 
    
    espacio += v
    tiempo  += 1

if len(tray_ideal) > 1000:
  print ("<!> Trayectoria muy larga - puede que no se haya alcanzado la posicion final.")
print ("Recorrido: "+str(round(espacio,3))+"m / "+str(tiempo/FPS)+"s")
print ("Distancia real al objetivo: "+\
    str(round(distancia(tray_real[-1],objetivos[-1]),3))+"m")
mostrar(objetivos,tray_ideal,tray_real)  # Representaciï¿½n grï¿½fica

