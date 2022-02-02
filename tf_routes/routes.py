import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import MCS
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from rdkit.Chem import *
IPythonConsole.ipython_useSVG=False  
import os
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw, ImageFont
import PIL
import numpy as np
from IPython.display import Image 
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
import math
from copy import deepcopy
from IPython.display import SVG
import time
import networkx as nx
import logging
logger = logging.getLogger(__name__)


# ## 4 different functions for calculating mutations (_calculate_order_of_LJ_mutations, _calculate_order_of_LJ_mutations_new, _calculate_order_of_LJ_mutations_iter and _calculate_order_of_LJ_mutations_iter_change):
# ### _calculate_order_of_LJ_mutations: naive dfs (as currently in transformato)
# ### _calculate_order_of_LJ_mutations_new: bfs/djikstra-algorithm applied once for route
# ### _calculate_order_of_LJ_mutations_new_iter: bfs/djikstra-algorithm applied iteratively, i.e. after each removal of an atom 
# ### _calculate_order_of_LJ_mutations_new_iter: works iteratively, i.e. after each removal of an atom, algorithm is chosen depending on current state

 
def _calculate_order_of_LJ_mutations(
    connected_dummy_regions: list, match_terminal_atoms: dict, G: nx.Graph
) -> list:

    ordered_LJ_mutations = []
    for real_atom in match_terminal_atoms:
        for dummy_atom in match_terminal_atoms[real_atom]:
            for connected_dummy_region in connected_dummy_regions:
                # stop at connected dummy region with specific dummy_atom in it
                if dummy_atom not in connected_dummy_region:
                    continue

                G_dummy = G.copy()
                # delete all nodes not in dummy region
                remove_nodes = [
                    node for node in G.nodes() if node not in connected_dummy_region
                ]
                for remove_node in remove_nodes:
                    G_dummy.remove_node(remove_node)

                # root is the dummy atom that connects the real region with the dummy region
                root = dummy_atom

                edges = list(nx.dfs_edges(G_dummy, source=root))
               #     logger.info("edges")
               #     logger.info(edges)
                nodes = [root] + [v for u, v in edges]
              #    logger.info("nodes")
              #    logger.info(nodes)
                nodes.reverse()  # NOTE: reverse the mutation
                ordered_LJ_mutations.append(nodes)

    return ordered_LJ_mutations

 
def _calculate_order_of_LJ_mutations_new(
    connected_dummy_regions: list, match_terminal_atoms: dict, G: nx.Graph, cyclecheck=True, ordercycles=True
) -> list:

    ordered_LJ_mutations = []
    
    for real_atom in match_terminal_atoms:
        for dummy_atom in match_terminal_atoms[real_atom]:
            for connected_dummy_region in connected_dummy_regions:
                # stop at connected dummy region with specific dummy_atom in it
                if dummy_atom not in connected_dummy_region:
                    continue

                G_dummy = G.copy()
                # delete all nodes not in dummy region
                remove_nodes = [
                    node for node in G.nodes() if node not in connected_dummy_region
                ]
                for remove_node in remove_nodes:
                    G_dummy.remove_node(remove_node)

                # root is the dummy atom that connects the real region with the dummy region
                root = dummy_atom
                
                #process cycles
                if (cyclecheck == True and ordercycles==False):
                    G_dummy = cycle_checks_nx(G_dummy)
                
                #process cycles and correct order (according to 'preferential removal')
                if (cyclecheck == True and ordercycles==True):
                    cycledict, degreedict = cycle_checks(G_dummy)
                             
                        
                #dijkstra          
                ssource = nx.single_source_dijkstra(G_dummy, source=root, weight='weight')
     
                sortedssource = {k: v for k, v in sorted(ssource[0].items(), key=lambda item: item[1])}   
    
                max_node = max(ssource[0], key=ssource[0].get)
                      
                sortedssource_edges = sortedssource.keys()

                sortedssource_edges_list = list(sortedssource_edges)
                 
                nodes = sortedssource_edges_list
            
                nodes.reverse()  
                
                #sort nodes according to degree, cycle participation and removal order
                if (cyclecheck == True and ordercycles==True):
                    nodes = change_route_cycles(nodes, cycledict, degreedict, sortedssource, G)
             
                
                logger.info("Final mutation route:")
                logger.info(nodes)
                ordered_LJ_mutations.append(nodes)

    return ordered_LJ_mutations

 
def _calculate_order_of_LJ_mutations_new_iter(
    connected_dummy_regions: list, match_terminal_atoms: dict, G: nx.Graph, cyclecheck=True, ordercheck=True
) -> list:

    ordered_LJ_mutations = []
    
    for real_atom in match_terminal_atoms:
        for dummy_atom in match_terminal_atoms[real_atom]:
            for connected_dummy_region in connected_dummy_regions:
                # stop at connected dummy region with specific dummy_atom in it
                if dummy_atom not in connected_dummy_region:
                    continue

                G_dummy = G.copy()
                # delete all nodes not in dummy region
                remove_nodes = [
                    node for node in G.nodes() if node not in connected_dummy_region
                ]
                for remove_node in remove_nodes:
                    G_dummy.remove_node(remove_node)

                # root is the dummy atom that connects the real region with the dummy region
                root = dummy_atom

           
                
                final_order = []
                
                removeG = nx.Graph()
                removearray = []
                while ( len(G_dummy.nodes())  > 0):
                    #update weights according to already removed nodes
                    if (ordercheck == True):
                        G_dummy = order_checks_nx(G_dummy, removearray, G)
                    
                    G_origweights = G_dummy.copy()
                    
                    #update weights according to cycle participation
                    if (cyclecheck == True):
                        G_dummy = cycle_checks_nx(G_dummy)
                   

                    
                    #dijkstra    
                    ssource = nx.single_source_dijkstra(G_dummy, source=root, weight='weight')

                    sortedssource = {k: v for k, v in sorted(ssource[0].items(), key=lambda item: item[1])}
                    max_node = max(ssource[0], key=ssource[0].get)
            
                    final_order.extend([max_node])
                    
                    
                    #restore original weights
                    G_dummy = G_origweights
                    
                    #remove G_dummy
                    G_dummy.remove_node(max_node)
                    
                    #add to removeG
                    removeG.add_node(max_node)
                    removearray.append(max_node)
                    
                    sortedssource_edges = final_order
                    
              
                #sortedssource_edges_list already contains the nodes in right (reversed) order (including root)
                nodes = sortedssource_edges
                
                logger.info("Final mutation route:")
                logger.info(nodes)
                
                #no reverse (already correct order, starting with the node with greatest distance from root)
                ordered_LJ_mutations.append(nodes)
                
               

    return ordered_LJ_mutations

 
def _calculate_order_of_LJ_mutations_new_iter_change(
    connected_dummy_regions: list, match_terminal_atoms: dict, G: nx.Graph, cyclecheck=True, ordercheck=True
) -> list:

    ordered_LJ_mutations = []

    initial_cdict, initial_degreedict = cycle_checks(G)
    
    cycles = nx.cycle_basis(G)
   

    for real_atom in match_terminal_atoms:
        for dummy_atom in match_terminal_atoms[real_atom]:
            for connected_dummy_region in connected_dummy_regions:
                # stop at connected dummy region with specific dummy_atom in it
                if dummy_atom not in connected_dummy_region:
                    continue

                G_dummy = G.copy()
                
                G_orig = G.copy()
                # delete all nodes not in dummy region
                remove_nodes = [
                    node for node in G.nodes() if node not in connected_dummy_region
                ]
                for remove_node in remove_nodes:
                    G_dummy.remove_node(remove_node)

                # root is the dummy atom that connects the real region with the dummy region
                root = dummy_atom

                dfs_step = False
                cycle_step = False
                cycle_step_initialized = False
                illegal_root_cycle = False
 
                final_order = []
                
                removeG = nx.Graph()
                removearray = []
                
                
                while ( len(G_dummy.nodes())  > 0):
                    
                    #update weights according to already removed nodes
                    if (ordercheck == True):
                        G_dummy = order_checks_nx(G_dummy, removearray, G)
                    
                    G_origweights = G_dummy.copy()
                    
                    #update weights according to cycle participation
                    if (cyclecheck == True):
                        G_dummy = cycle_checks_nx(G_dummy)
                        

                    if (dfs_step == False and cycle_step == False):
                        #logger.info("dijkstra step")               
                        #dijkstra    
                        ssource = nx.single_source_dijkstra(G_dummy, source=root, weight='weight')

                        sortedssource = {k: v for k, v in sorted(ssource[0].items(), key=lambda item: item[1])}
                        max_node = max(ssource[0], key=ssource[0].get)

                        current_degree = G_dummy.degree(max_node)
                        
                        neighbors = [n for n in G_dummy.neighbors(max_node)]
                        
                        if (current_degree > 0):
                            current_neighbor = neighbors[0]
                        else:
                            current_neighbor = None
                            
                       
                                    
                        
                        if (current_neighbor != None):
                            if (initial_cdict[max_node] == 0 and current_degree == 1 and G_dummy.degree(current_neighbor) == 2):                     
                                dfs_step = True
                            
                            else:
                                illegal_root_cycle = False
                                
                                for array in cycles:
                                    if max_node in array and root in array:
                                        illegal_root_cycle = True
                     
                                
                                if (initial_cdict[current_neighbor] > 0 and illegal_root_cycle == False):
                                
                                    cycle_step = True
                                    cycle_step_initialized=False
                                  
                                    cyclepath = sortedssource
                                    dfs_step = False
                            
                            if (current_neighbor == root):
                                dfs_step = False
                    
                    elif (cycle_step == True):
                        #logger.info("cycle step")
                        if (cycle_step_initialized == False):
                           
                          
                        #get neighbors in cycle    
                            for array in cycles:
                                if current_neighbor in array:
                                    currentcyclenodes = array
                                    
                         
                            if max_node in currentcyclenodes:
                                currentcyclenodes.remove(max_node)
                         
                            cyclepath2 = []
                            for el in cyclepath:
                                if el in currentcyclenodes:
                                    cyclepath2.append(el)
 
                           #check current state (cycle participation, degree)
                            current_cdict, current_degreedict = cycle_checks(G_dummy)

                            cyclepath_final = []
                            
                            #check if the common core is correct, i.e. root does not participate in cycle
                            illegal_root_cycle = False
                            for node in currentcyclenodes:
                                if node not in connected_dummy_region:
                                    illegal_root_cycle = True
                                    dfs_step = False
                                    cycle_step = False
                                    cycle_step_initialized = False
                                    G_dummy = G_origweights.copy()
                                    continue
                                    
                            
                            if (len(cyclepath2) == 0):
                                dfs_step = False
                                cycle_step = False
                                cycle_step_initialized = False
                                G_dummy = G_origweights.copy()
                                continue
                                
                            
                            min_degree = current_degreedict[cyclepath2[0]]
                            for el in cyclepath2:
                                if (current_degreedict[el] < min_degree):
                                    min_degree = current_degreedict[el]
                        
                            #if degree is too high, switch to dijkstra
                            if (min_degree>1):
                                dfs_step = False
                                cycle_step = False
                                cycle_step_initialized = False
                                G_dummy = G_origweights.copy()
                                continue
                                
                            for el in cyclepath2:
                                if (current_degreedict[el] == min_degree):
                                    cyclepath_final.append(el)

                            cycle_step_initialized = True
                       
                        
                        problematic_neighbors = False
                        neighbors = [n for n in G_dummy.neighbors(cyclepath_final[-1])]
                        
                     
                        
                        if (len(neighbors) > 0):
                            for neighbor in neighbors:
                                
                                if (initial_cdict[neighbor]== 0 or initial_cdict[max_node]== 0):
                                    problematic_neighbors = True
                                   
                                    continue
                                #only for pathological cases (pdbbind l1a4k)
                                if ((initial_cdict[max_node] > 2 or initial_cdict[cyclepath_final[-1]] > 2) and G.degree(neighbor) < 3):
                                    problematic_neighbors = True
                                 
                                    continue
                                    
                        for elem in cyclepath:
                            if (G.degree(elem) > 3):
                                problematic_neighbors = True

                        
                        if (problematic_neighbors == True):
                          
                            dfs_step = False
                            cycle_step = False
                            cycle_step_initialized = False
                            #restore
                            G_dummy = G_origweights.copy()
                            continue
                                        
                             
                        #only for pathological cases (root pertains to a cycle)
                        if root in cyclepath_final:
                            dfs_step = False
                            cycle_step = False
                            cycle_step_initialized = False
                            G_dummy = G_origweights.copy()
                            continue

                        illegal_root_cycle = False
                    
                        for array in cycles:
                            if (current_neighbor in array or max_node in array or cyclepath_final[-1] in array) and root in array:
                                
                                illegal_root_cycle = True
                                dfs_step = False
                                cycle_step = False
                                cycle_step_initialized = False
                                continue
                        if (illegal_root_cycle == True):
                            #restore
                            G_dummy = G_origweights.copy()
                            continue
                                
                        max_node = cyclepath_final.pop()
                        
                        
                        if (len(cyclepath_final) == 0):
                            dfs_step = False
                            cycle_step = False
                            cycle_step_initialized = False
                        
  
                    else:
                        #logger.info("dfs step")
                        max_node = current_neighbor
                        
                    
                        current_degree = G_dummy.degree(max_node)
                      
                        if (current_degree > 0):
                            current_neighbor = [n for n in G_dummy.neighbors(max_node)][0]
                        else:
                            current_neighbor = None
                       
                        if (current_neighbor != None):
                            if (initial_cdict[max_node] > 0 or current_degree != 1 or G_dummy.degree(current_neighbor) != 2):                      
                                dfs_step = False 
                            current_cdict, current_degreedict = cycle_checks(G_dummy)
                            if (current_cdict[max_node] > 0):
                                ssource = nx.single_source_dijkstra(G_dummy, source=root, weight='weight')
                                sortedssource = {k: v for k, v in sorted(ssource[0].items(), key=lambda item: item[1])}
                                cyclepath = sortedssource
                                cycle_step = True
                        if (current_neighbor == root):
                            dfs_step = False
                 
                
                    if (max_node == root and len(G_dummy.nodes())  > 1):
                        illegal_root_cycle = True
                        dfs_step = False
                        cycle_step = False
                        cycle_step_initialized = False
                        G_dummy = G_origweights.copy()
                        
                        continue
                        
                    final_order.extend([max_node])

                    G_dummy = G_origweights.copy()

                    #remove G_dummy
                    G_dummy.remove_node(max_node)

                    #add to removeG
                    removeG.add_node(max_node)
                    removearray.append(max_node)
                    
                  

                sortedssource_edges = final_order
                    
              
                #sortedssource_edges_list already contains the nodes in right (reversed) order (including root)
                nodes = sortedssource_edges
                
                logger.info("Final mutation route:")
                logger.info(nodes)
                
                #no reverse (already correct order, starting with the node with greatest distance from root)
                ordered_LJ_mutations.append(nodes)
                
               

    return ordered_LJ_mutations

    
# ### additional functions for the new mutation route algorithms (using weight updates, cycle and degree functions of networkx): process cycles (atoms participating in many cycles are removed later) and 'preferential removal' (atoms which neighbours already have been removed are removed earlier)

 
#cycle processing, currently used in _calculate_order_of_LJ_mutations_new_iter 
def cycle_checks_nx(G):
    
    #search cycles using networkx
    cycles = nx.cycle_basis(G)
    
    import collections
    from collections import Counter
    
    cdict = Counter(x for xs in cycles for x in set(xs))
  
    
    #modify weighted graph: nodes participating in many cycles get lower weight
    for i in cdict:
        edg = G.edges(i)
        for el in edg:
            G[el[0]][el[1]]['weight'] = G[el[0]][el[1]]['weight'] - cdict[i]*5
    
    return G

 
#preferential removal, currently used in _calculate_order_of_LJ_mutations_new_iter 
def order_checks_nx(G, removearray, G_total):
    if (len(removearray) > 0):
        lastremoved = removearray[len(removearray) - 1]
        
        edg = G_total.edges(lastremoved)
        
        edg_dummy = G.edges()
        
        for ed in edg:
       
            if (ed[0]!=lastremoved):
                connectednode = ed[0]
            else:
                connectednode = ed[1]
            
            #if node is connected to last removed node, its weight get a small increase
            if (G.has_node(connectednode)):
             
                connectededges = G.edges(connectednode)
                for el in connectededges:
                   
                    G[el[0]][el[1]]['weight'] = G[el[0]][el[1]]['weight'] + 1
                                    
            
    return G

 
#cycle processing dictionary and degree dictionary for preferential removal, currently used in _calculate_order_of_LJ_mutations_new (via change_route_cycles)
def cycle_checks(G):
    
    
    #search cycles using networkx
    cycles = nx.cycle_basis(G)
    
    #alternatively, using rdkit
    #ri = mol.GetRingInfo()
    #cyclesrdkit = ri.AtomRings()
    
    import collections
    from collections import Counter
    
    cdict = Counter(x for xs in cycles for x in set(xs))
    #cdictrdkit = Counter(x for xs in cyclesrdkit for x in set(xs))
    
    #add atoms with no cycle participation
    for key in G.nodes:
        if key not in cdict:
            cdict[key] = 0
    
    degreedict = G.degree()
    degreedict = {node:val for (node, val) in degreedict}      
    
    return cdict, degreedict

 
#currently used in _calculate_order_of_LJ_mutations_new
#preliminary mutation is list is sorted using cycle and degree dictionary
def change_route_cycles(route, cycledict, degreedict, weightdict, G):

    
    for i in range(len(route)-1):
        routedict = route[i]
        routeweight = weightdict.get(route[i])
        
        routecycleval = cycledict.get(route[i])
        routedegreeval = degreedict.get(route[i])
       
        for j in range(i, len(route)):
       
            if (routeweight == weightdict[route[j]]):
                      
                #if nodes have same weight (i.e. distance from root), the node participating in more cycles is removed later
                
                if (routecycleval > cycledict[route[j]] or (routecycleval == cycledict[route[j]] and routedegreeval > degreedict[route[j]])):
                    idx1 = route.index(route[i])
                    idx2 = route.index(route[j])                  
                    route[idx1], route[idx2] = route[idx2], route[idx1]                 
                    continue
                    
                #if nodes have same weight (i.e. distance from root) and same cycle participation number, the node which has more neighbours already removed is removed earlier

                
                if (routecycleval == cycledict[route[j]]):
                    
                    edgesi = G.edges(routedict)
                    edgesj = G.edges(route[j])

                    iedgecounter = 0
                    for edge in edgesi:
                        if (edge[1] in route[0:i] or edge[0] in route[0:i] ):

                            iedgecounter = iedgecounter + 1

                    jedgecounter = 0
                    for edge in edgesj:
                        if (edge[1] in route[0:i] or edge[0] in route[0:i]):

                            jedgecounter = jedgecounter + 1


                    if (iedgecounter < jedgecounter):
                        idx1 = route.index(route[i])
                        idx2 = route.index(route[j])
                        route[idx1], route[idx2] = route[idx2], route[idx1]

    
    return route