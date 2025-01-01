#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional - 
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs
import math
import json

# ******************************************************************************
# Declaración de funciones

def muestra_origenes(O,final=0):
  # Muestra los orígenes de coordenadas para cada articulación
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O'+str(i)+')0\t= '+str([round(j,3) for j in O[i]]))
  if final:
    print('E.Final = '+str([round(j,3) for j in final]))

def muestra_robot(O,obj):
  # Muestra el robot graficamente
  plt.figure()
  plt.xlim(-L,L)
  plt.ylim(-L,L)
  T = [np.array(o).T.tolist() for o in O]
  for i in range(len(T)):
    plt.plot(T[i][0], T[i][1], '-o', color=cs.hsv_to_rgb(i/float(len(T)),1,1))
  plt.plot(obj[0], obj[1], '*')
 # plt.pause(1.0)
  plt.show()
  
  input()
  plt.close()

def matriz_T(d,th,a,al):
   
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)]
         ,[sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)]
         ,[      0,          sin(al),          cos(al),         d]
         ,[      0,                0,                0,         1]
         ]

def cin_dir(th,a):
  #Sea 'th' el vector de thetas
  #Sea 'a'  el vector de longitudes
  T = np.identity(4)
  o = [[0,0]]
  for i in range(len(th)):
    T = np.dot(T,matriz_T(0,th[i],a[i],0))
    tmp=np.dot(T,[0,0,0,1])
    o.append([tmp[0],tmp[1]])
  return o

# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# Variables del manipulador

art_types = [] # Tipos de articulación; 1 = prismatica, 0 = rotacion
# valores articulares arbitrarios para la cinemática directa inicial
th=[]
a =[]
# límites superiores e inferiores
limit_sup=[]
limit_inf=[]

# Lectura de fichero
with open("robot.json", "r") as file:
    data = json.load(file)
    
for articulacion in data:
    if (articulacion['type'] == "rotacion"):
      art_types.append(0)
    elif (articulacion['type'] == "prismatica"):
      art_types.append(1)
    th.append(articulacion['th'])
    a.append(articulacion['a'])
    limit_sup.append(articulacion['limit_sup'])
    limit_inf.append(articulacion['limit_inf'])

L = sum(a) # variable para representación gráfica
EPSILON = .1

plt.ion() # modo interactivo

# introducción del punto para la cinemática inversa
if len(sys.argv) != 3:
  sys.exit("python " + sys.argv[0] + " x y")
objetivo=[float(i) for i in sys.argv[1:]]
O=cin_dir(th,a)
#O=zeros(len(th)+1) # Reservamos estructura en memoria
 # Calculamos la posicion inicial
print ("- Posicion inicial:")
muestra_origenes(O)

dist = float("inf")
prev = 0.
iteracion = 1
while (dist > EPSILON and abs(prev-dist) > EPSILON/100.):
  prev = dist
  O=[cin_dir(th,a)]
  
  # Para cada combinación de articulaciones:
  for i in range(len(th)):
    indice_actuador = len(th)-(i+1)
    EF = O[i][len(th)] # Punto final del brazo; Actualizacion con los calculos de cinematica directa de cada iteración
    
    # cálculo de la cinemática inversa:
    # O[0] son las posiciones de cada origen antes de la primera iteracion
    # alinear O[0][len(th)-(i+2)], O[0][len(th)-(i+1)]
    print("objetivo", [x for x in objetivo])
    o_rotacion = O[0][len(th)-(i+1)]
    print("O-1", [x for x in o_rotacion])
   
    if (art_types[indice_actuador]): # Articulación prismática
      w = 0
      for x in range(i):
        w += th[x]
        
      u = [math.cos(w), math.sin(w)]
      v = [objetivo[0]-EF[0], objetivo[1]-EF[1]]
      d = np.dot(u,v)
      
      print ("d", d);
      new_a = a[len(th)-(i+1)] + d
      
      # límites
      if (new_a < limit_inf[indice_actuador]):
          new_a = limit_inf[indice_actuador]
      elif (new_a > limit_sup[indice_actuador]):
          new_a = limit_sup[indice_actuador]
          
      a[len(th)-(i+1)] = new_a 

    else:
      # vector del punto de rotacion al punto objetivo
      w =  [a - b for a, b in zip(objetivo, o_rotacion)]
      w_magnitude  = sqrt( w[0]**2+ w[1]**2)
      w = [x/w_magnitude for x in w]
      
      # vector del punto de rotacion al punto extremo EF
      v = [a - b for a, b in zip(EF, o_rotacion)]
      v_magnitude  = sqrt( v[0]**2+ v[1]**2)
      v = [x/v_magnitude for x in v]

      # arcotangente
      alpha1 = math.atan2(w[1], w[0])
      alpha2 = math.atan2(v[1], v[0])
      
      delta = alpha1-alpha2
      new_th = delta + th[len(th)-(i+1)]
      
      # corrección del ángulo (-pi <= delta <= pi)
      while new_th > math.pi:
        new_th -= 2*math.pi
      while new_th < -math.pi:
        new_th += 2*math.pi
        
      # límites
      if (new_th < limit_inf[indice_actuador]):
        new_th = limit_inf[indice_actuador]
      elif (new_th > limit_sup[indice_actuador]):
        new_th = limit_sup[indice_actuador]
        
      th[len(th)-(i+1)] = new_th # actualizamos th

    # Calculamos cinemática directa  
    O.append(cin_dir(th,a))

  dist = np.linalg.norm(np.subtract(objetivo,O[-1][-1]))
  print ("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(O[-1])
  muestra_robot(O,objetivo)
  print ("Distancia al objetivo = " + str(round(dist,5)))
  iteracion+=1
  O[0]=O[-1]

if dist <= EPSILON:
  print ("\n" + str(iteracion) + " iteraciones para converger.")
else:
  print ("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
print ("- Umbral de convergencia epsilon: " + str(EPSILON))
print ("- Distancia al objetivo:          " + str(round(dist,5)))
print ("- Valores finales de las articulaciones:")
for i in range(len(th)):
  print ("  theta" + str(i+1) + " = " + str(round(th[i],3)))
for i in range(len(th)):
  print ("  L" + str(i+1) + "     = " + str(round(a[i],3)))
