from ovito.io import import_file
from ovito.io import export_file
from ovito.modifiers import *
#import ovito.data.nearest_neighbor_finder
import ovito
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from random import uniform
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist

def calculateVoidsScy(frame, data):

    voronoiVertices = data.surfaces['voronoi-polyhedra'].get_vertices()
    atomPos = data.particles['Position']
    verticesVoidsx = np.zeros(3*len(voronoiVertices))
    verticesVoidsy = np.zeros(3*len(voronoiVertices))
    verticesVoidsz = np.zeros(3*len(voronoiVertices))
    cavityVoids = np.zeros(3*len(voronoiVertices))
    k = -1

    start = time.time()

    print('List from scypY:')
    for i,atom in enumerate(atomPos):
        if len(atomPos) > 100:
            print('Too many particles to calculate. Select less particles or increase the limit via code.')
            break

        cavityT = data.particles['Cavity Radius'][i]
        #print('particles:',i, atom, ' cavity radius:', cavityT )

        #monster code that calculates the distacances of a particle with a vector of particles
        distances = cdist([atom], voronoiVertices, 'euclidean')

        index = np.where( distances == cavityT)
        #print('index is ',index)

        if(index[1].size > 0 ):
            k=k+1

            verticesVoidsx[k]= voronoiVertices[index[1]][0][0]
            verticesVoidsy[k]= voronoiVertices[index[1]][0][1]
            verticesVoidsz[k]= voronoiVertices[index[1]][0][2]
            cavityVoids[k]=cavityT
    print('Time spent on scypy:',time.time()-start)

    vunique = set(zip(verticesVoidsx,verticesVoidsy,verticesVoidsz, cavityVoids))

    print(len(vunique)-1)
    #print('# type, x,y,z, cavity')
    for i,ele in enumerate(vunique):
        if(ele[2]!=0):
            data.particles_.create_particle((ele[0],ele[1], ele[2]))
            data.particles_.create_property('Radius')[-1] = ele[3]
            data.particles_.create_property('Transparency')[-1] = 0.1


def calculateVoids(frame, data):

    voronoiVertices = data.surfaces['voronoi-polyhedra'].get_vertices()
    atomPos = data.particles['Position']
    verticesVoidsx = np.zeros(3*len(voronoiVertices))   #should be 3 not 4
    verticesVoidsy = np.zeros(3*len(voronoiVertices))
    verticesVoidsz = np.zeros(3*len(voronoiVertices))
    cavityVoids = np.zeros(3*len(voronoiVertices))
    k = -1

    start = time.time()
    for i,atom in enumerate(atomPos):
        if len(atomPos) > 100:
            print('We have more than 100 particles selected, processe will be terminated to avoid crash. Remove this line from script and add an yield')
            break
        cavityT = data.particles['Cavity Radius'][i]
        #print('particles:',i, atom, ' cavity radius:', cavityT )
        #yield
        for j,vertex in enumerate(voronoiVertices):
            #distMath might be faster
            distT = math.dist(atom, vertex)

            #print('    vertex',j, vertex, ' distance:', distT)
            #if (%( distT < 4.4)% and (distT == cavityT)):
            if (distT == cavityT*ovitoFree): #((distT - cavityT) < 10E-8  ): # (distT == cavityT*ovitoFree)):

                #start2 = time.time()
                k=k+1
                #print('    vertex',j, vertex)
                verticesVoidsx[k]=vertex[0]
                verticesVoidsy[k]=vertex[1]
                verticesVoidsz[k]=vertex[2]
                cavityVoids[k]=cavityT*ovitoFree

                #end2 = time.time()
                #print('Append time:', end2-start2)
    end = time.time()
    print('Time spent:',end-start)


    vunique = set(zip(verticesVoidsx,verticesVoidsy,verticesVoidsz, cavityVoids))
    #print(vunique)

    #print(zip(*vunique))
    #print(len(vunique)-1)
    #print('# type, x,y,z, cavity')
    for i,ele in enumerate(vunique):
        if(ele[2]!=0):
            #print('H ',ele[0],ele[1], ele[2],ele[3] )
            data.particles_.create_particle((ele[0],ele[1], ele[2]))
            data.particles_.create_property('Radius')[-1] = ele[3]
            data.particles_.create_property('Transparency')[-1] = 0.1

def selectCarbons(frame: int, data):

    # This user-defined modifier function gets automatically called by OVITO whenever the data pipeline is newly computed.
    # It receives two arguments from the pipeline system:
    #
    #    frame - The current animation frame number at which the pipeline is being evaluated.
    #    data  - The DataCollection passed in from the pipeline system.
    #            The function may modify the data stored in this DataCollection as needed.
    #
    # What follows is an example code snippet doing nothing aside from printing the current
    # list of particle properties to the log window. Use it as a starting point for developing
    # your own data modification or analysis functions.

    if data.particles != None:
        print("There are %i particles with the following properties:" % data.particles.count)
        for property_name in data.particles.keys():
            print("  '%s'" % property_name)


    #data.particles_.create_property("Selection", data = numpy.isin(data.particles.identifiers, list))
    data.apply(ExpressionSelectionModifier(expression = "ParticleType == 2"))

    #Prefetch the particle property Selection
    selection = data.particles.selection
    #Indices of the selected atoms
    selected_atoms = np.where(selection == 1)[0]

    x0C = data.particles.positions[selected_atoms[0]][0]
    y0C = data.particles.positions[selected_atoms[0]][1]
    z0C = data.particles.positions[selected_atoms[0]][2]
    print('POsition of the carbon atom x,y,z:', x0C,y0C,z0C)

    #for atom in selected_atoms:
    #    print(data.particles.positions[atom])

    #select 1 solvation shell around the carbon atom
    data.apply(ExpressionSelectionModifier(expression = '(((Position.X - '+str(x0C)+')^2 + (Position.Y -'+str(y0C)+')^2 + (Position.Z -'+str(z0C)+')^2) < (4.45)^2) &&  (ParticleType == 5) || (ParticleType == 2)'))

    #delete the other particles
    data.apply(InvertSelectionModifier())
    data.apply(DeleteSelectedModifier())

    #print(data.particles.positions.array)

def selectCarbonsNNN(frame: int, data):

    # This user-defined modifier function gets automatically called by OVITO whenever the data pipeline is newly computed.
    # It receives two arguments from the pipeline system:
    #
    #    frame - The current animation frame number at which the pipeline is being evaluated.
    #    data  - The DataCollection passed in from the pipeline system.
    #            The function may modify the data stored in this DataCollection as needed.
    #
    # What follows is an example code snippet doing nothing aside from printing the current
    # list of particle properties to the log window. Use it as a starting point for developing
    # your own data modification or analysis functions.

    #if data.particles != None:
    #    print("There are %i particles with the following properties:" % data.particles.count)
    #    for property_name in data.particles.keys():
    #        print("  '%s'" % property_name)


    #data.particles_.create_property("Selection", data = numpy.isin(data.particles.identifiers, list))
    data.apply(ExpressionSelectionModifier(expression = "ParticleType == 2"))

    #Prefetch the particle property Selection
    selection = data.particles.selection
    #Indices of the selected atoms
    selected_atoms = np.where(selection == 1)[0]

    x0C = data.particles.positions[selected_atoms[0]][0]
    y0C = data.particles.positions[selected_atoms[0]][1]
    z0C = data.particles.positions[selected_atoms[0]][2]
    #print('Position of the carbon atom x,y,z:', x0C,y0C,z0C, ' atom selected is: ', selected_atoms)

    NselectedOxygens = 0
    theHatefulEight = np.zeros(8, dtype=int)  #will have the 8 water that represent the first solvation shell
    searchNumber = 16

    while (NselectedOxygens < 8):
        searchNumber += 1
        NselectedOxygens = 0

        # Set up a neighbor finder for visiting the 12 closest neighbors of each particle.
        finder = ovito.data.NearestNeighborFinder(searchNumber, data)

        # Visit particles closest to some particle (in this case the selected carbon)
        for neigh in finder.find(selected_atoms):
            if (data.particles.particle_types[neigh.index] == 5): #5 is for oxygens
                #print(neigh.index, neigh.distance,  data.particles.positions[neigh.index], data.particles.particle_types[neigh.index])
                theHatefulEight[NselectedOxygens]=neigh.index
                NselectedOxygens += 1


    #reset selection and selected the 8 oxygens
    selection[:]=0
    selection[theHatefulEight] = 1

    #this method will alter the data.particles.selection directly, and leave selection unaltered
    data.apply(ExpandSelectionModifier(cutoff=1.6))

    #Check the particles that were selected
    #print('after selection from data : ', np.where(data.particles.selection == 1))
    #for item in np.where(data.particles.selection == 1):
    #    print(item,' ', data.particles.particle_types[item])


    #delete the other particles
    data.apply(InvertSelectionModifier())
    data.apply(DeleteSelectedModifier())

    #print(data.particles.positions.array)


def computeVolumes(frame, data):
    # if data.particles != None:
    #    print("There are %i particles with the following properties:" % data.particles.count)
    #    for property_name in data.particles.keys():
    #        print("  '%s'" % property_name)

    ovitoFree = 1

    # how does it know that this is just the undeleted particles?
    data.apply(SelectTypeModifier(property='Particle Type', types={0}))

    # Prefetch the particle property Selection
    selection = data.particles.selection
    # Indices of the selected atoms
    selected_atoms = np.where(selection == 1)[0]

    voidPositions = data.particles.positions[selected_atoms]
    # voidCavity = data.particles['Radius']
    # print(voidPositions)

    voidList = []

    for ele in selected_atoms:
        voidList.append((data.particles.positions[ele][0], data.particles.positions[ele][1],
                         data.particles.positions[ele][2], data.particles['Radius'][ele]))
        # print(voidList[-1])
        # print(ele)

    # do the volumes calculation
    Nsamples = 10000000

    # calculates the center of mass of the system
    # will be used as a reference for the reduced cell
    # from which the random point will be sample using Monte Carlo
    cm = np.sum(voidPositions, axis=0)
    if len(voidPositions) == 0:
        cm = [0, 0, 0]
    else:
        cm /= len(voidPositions)
    # print('CM:', cm)

    # maxRange = np.max(voidPositions+np.sign(voidPositions)*voidRadius, axis=0)-np.min(voidPositions+np.sign(voidPositions)*voidRadius, axis=0)
    maxRange = [4, 4, 4]

    if (len(voidPositions) == 1):  # this is the single void case
        maxRange[0] = voidList[-1][3]
        maxRange[1] = voidList[-1][3]
        maxRange[2] = voidList[-1][3]
    # print('Max range:',maxRange)

    # data.particles_.create_particle([0,0,0])
    f = 2.0
    data.particles_.create_particle([cm[0] - f * maxRange[0], cm[1] - f * maxRange[1], cm[2] - f * maxRange[2]])
    data.particles_.create_particle([cm[0] - f * maxRange[0], cm[1] + f * maxRange[1], cm[2] - f * maxRange[2]])
    data.particles_.create_particle([cm[0] + f * maxRange[0], cm[1] - f * maxRange[1], cm[2] - f * maxRange[2]])
    data.particles_.create_particle([cm[0] + f * maxRange[0], cm[1] + f * maxRange[1], cm[2] - f * maxRange[2]])

    data.particles_.create_particle([cm[0] - f * maxRange[0], cm[1] - f * maxRange[1], cm[2] + f * maxRange[2]])
    data.particles_.create_particle([cm[0] - f * maxRange[0], cm[1] + f * maxRange[1], cm[2] + f * maxRange[2]])
    data.particles_.create_particle([cm[0] + f * maxRange[0], cm[1] - f * maxRange[1], cm[2] + f * maxRange[2]])
    data.particles_.create_particle([cm[0] + f * maxRange[0], cm[1] + f * maxRange[1], cm[2] + f * maxRange[2]])

    # these are the reduced cell volume (samples cell)
    # why not 2 max range?
    cellX = 4 * maxRange[0]
    cellY = 4 * maxRange[1]
    cellZ = 4 * maxRange[2]

    countInsideVolume = 0
    countOutsideVolume = 0
    distance = 0
    flag = False
    N = 0
    Nsamples = 100000

    ovitoFree = 1

    # print('Calculating:',frame)
    start = time.time()
    while (N < Nsamples):
        N = N + 1

        # generate a random number in the reducedCell (samples cell), that the monte Carlo will samples
        # if original large cell is used, the monte Carlo will need more points to converge.
        xr = uniform(cm[0] - 2 * maxRange[0], cm[0] + 2 * maxRange[0])
        yr = uniform(cm[1] - 2 * maxRange[1], cm[1] + 2 * maxRange[1])
        zr = uniform(cm[2] - 2 * maxRange[2], cm[2] + 2 * maxRange[2])

        # fsamplesM.write(str(xr)+' '+str(yr)+' '+ str(zr)+'\n')
        flag = False

        for i, ele in enumerate(voidList):
            distance = math.dist((xr, yr, zr), (ele[0], ele[1], ele[2]))

            # !!!!!!!!!
            if abs(distance) <= (ele[
                                     3] * ovitoFree):  # super care here, ovito free, gives the cavity diameter instead of cavity radius
                # print('\n point inside ',x,y,z,' is inside circle',i,
                #      ' ',(ele[0],ele[1],ele[2]), ' with radius: ',ele[3],
                #     ' dist:',distance)
                flag = True
                break  # insure that the point is not counted twice, as the sphere can overlap
        if flag == False:
            countOutsideVolume += 1

        else:
            countInsideVolume += 1

    end = time.time()
    # print('Time elapsed:',end-start)
    # print('Points inside:', countInsideVolume)
    # print('Points outside:', countOutsideVolume)

    TotalVolume = cellX * cellY * cellZ
    volumeInside = TotalVolume * (countInsideVolume / (countInsideVolume + countOutsideVolume))
    volumeOutside = TotalVolume * (countOutsideVolume / (countInsideVolume + countOutsideVolume))
    # print('Total volume:', TotalVolume )
    # print('Volume outside:', volumeOutside)
    # print('\nVolume inside:', volumeInside, ' ',100*countInsideVolume/(countInsideVolume+countOutsideVolume),'%')

    # check what is the volume occupied by the spheres if they were separated
    allSpheresVolume = 0
    for i, ele in enumerate(voidList):
        allSpheresVolume += (4 / 3) * math.pi * (ovitoFree * ovitoFree * ovitoFree * ele[3] * ele[3] * ele[3])
    # print('Isolated sphere volume: ', allSpheresVolume)

    return (volumeInside, len(voidList))

ovitoFree = 1



#if __name__ == "__main__":

#read the MD file

pipeline = import_file('MDanalysis-voidsWaterOnly.xyz')
data0=pipeline.compute()

Nframes = pipeline.source.num_frames
Nframes = 10
print('Total number of frames:', Nframes)

print('Begining calculation...wait..')

frames = range(0,Nframes,1)


voidVolumes = np.empty(Nframes)
sphereNumber = np.empty(Nframes)
#read the MD file


pipeline = import_file('./SCNtraj/MD/SCN-MD-2ns-centered-800.xyz')
#print('PBC?',pipeline.source.data.cell_.pbc)
#pipeline.source.data.cell_.pbc = (False,False,False)
#print('PBC?',pipeline.source.data.cell_.pbc)


#this is needed for voronoi analysis
mod = AffineTransformationModifier(
        operate_on = {'cell'}, # Transform box but not the particles.
        transformation = [[1.05, 0, 0, 0],
                          [0, 1.05, 0, 0],
                          [0, 0, 1.05, 0]])
pipeline.modifiers.append(mod)

#data.cell_.pbc = (True,True,True)

# Set up the Voronoi analysis modifier.
voro = VoronoiAnalysisModifier(
    compute_indices = True,
    use_radii = True,
    edge_threshold = 0.1,
    generate_polyhedra = True
)
pipeline.modifiers.append(voro)
data = pipeline.compute()

#
#           compute volumes
#

Nframes = pipeline.source.num_frames
#Nframes = 10

print('Total number of frames:', Nframes)
voidVolumes = np.empty(Nframes)
sphereNumber = np.empty(Nframes)

start = time.time()
print('Calculating volume..')
for frame in range(0,Nframes,1):
    data = pipeline.compute(frame)
    selectCarbons(frame, data)
    #calculateVoids(frame, data)
    #voidVolumes[frame],sphereNumber[frame]  = computeVolumes(frame, data)
    #selectCarbonsNNN(frame, data)
    #calculateVoids(frame, data)
    calculateVoidsScy(frame, data)
    voidVolumes[frame],sphereNumber[frame]  = computeVolumes(frame, data)
print('Time elapsed:',time.time() - start)

plt.plot(voidVolumes)
plt.show()
plt.hist(voidVolumes)
plt.show()
plt.plot(sphereNumber)
plt.show()

print('Mean volume:',voidVolumes.mean())
print('standart deviation:',voidVolumes.std())
print('Mean #sphere', sphereNumber.mean())
print('Mean #sphere deviation:',sphereNumber.std())




#from multiprocess import Pool
import multiprocessing as mp

if __name__ == '__main__':
    print('Cpu count:',mp.cpu_count())

    print('Calculating volume parallel:')
    start = time.time()

    #divide between 4 processors
    with mp.Pool(4) as pool:
        results = pool.map(taskAdvanced, frames)

    print('Elapsed time for ', Nframes,' was ', time.time()-start,' s')


    for i,r in enumerate(results):
        #print(r)
        voidVolumes[i],sphereNumber[i] = r
    plt.plot(voidVolumes)
    plt.show()
    plt.plot(sphereNumber)
    plt.show()

    print('Mean volume:', voidVolumes.mean())
    print('standart deviation:', voidVolumes.std())
    print('Mean #sphere', sphereNumber.mean())