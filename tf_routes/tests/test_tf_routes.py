"""
Unit and regression test for the tf_routes package.
"""

# Import package, test suite, and other packages as needed
import sys
import rdkit

from rdkit import Chem

import pytest

from copy import deepcopy

import tf_routes
from tf_routes import preprocessing
from tf_routes import routes
from tf_routes import visualizations

def test_tf_routes_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "tf_routes" in sys.modules

def test_routes_compare():
    smiless = ["CC1=CC2=CC=CC=C2N1", "CC(C)(C)C", "CC1=CC=CC=C1", "CC", "CC1=CC=CO1"]
    
    molarray1 = []
    for smile in smiless:
        mol = Chem.MolFromSmiles(smile)
        molarray1.append(mol)
    molarray2 = molarray1

    for mol1 in molarray1:
        for mol2 in molarray2:

            mol1 = preprocessing.generate_apply_dicts(mol1)
            mol2 = preprocessing.generate_apply_dicts(mol2)

            mol1coreindex, mol2coreindex, hit_ats1, hit_ats2 = preprocessing.get_common_core(mol1, mol2)

            graphmol1 = preprocessing._mol_to_nx_full_weight(mol1)
            graphmol2 = preprocessing._mol_to_nx_full_weight(mol2)

            subg1, G_dummy1 = preprocessing._find_connected_dummy_regions_mol(mol1, mol1coreindex, graphmol1)
            subg2, G_dummy2 = preprocessing._find_connected_dummy_regions_mol(mol2, mol2coreindex, graphmol2)

            terminaldummy1, terminalreal1 = preprocessing._find_terminal_atom(mol1coreindex,  mol1)
            terminaldummy2, terminalreal2 = preprocessing._find_terminal_atom(mol2coreindex,  mol2)

            matchterminal1 = preprocessing._match_terminal_real_and_dummy_atoms(mol1, terminalreal1, terminaldummy1)
            matchterminal2 = preprocessing._match_terminal_real_and_dummy_atoms(mol2, terminalreal2, terminaldummy2)

            order1 = routes._calculate_order_of_LJ_mutations(
            subg1, matchterminal1, graphmol1
        )
            order2 = routes._calculate_order_of_LJ_mutations(
            subg2, matchterminal2, graphmol2
        )
            

            #new mutation order function - code with iteration
            order1new = routes._calculate_order_of_LJ_mutations_new(
            subg1, matchterminal1, graphmol1
        )
            order2new = routes._calculate_order_of_LJ_mutations_new(
            subg2, matchterminal2, graphmol2
        )
            
            #new mutation order function - code with iteration and different steps according to state
            order1newiter = routes._calculate_order_of_LJ_mutations_new_iter(
            subg1, matchterminal1, graphmol1
        )
            order2newiter = routes._calculate_order_of_LJ_mutations_new_iter(
            subg2, matchterminal2, graphmol2
        )

        #new mutation order function - code with iteration and different steps according to state
            order1newiter_change = routes._calculate_order_of_LJ_mutations_new_iter_change(
            subg1, matchterminal1, graphmol1
        )
            order2newiter_change = routes._calculate_order_of_LJ_mutations_new_iter_change(
            subg2, matchterminal2, graphmol2
        )

    

            sortorder1 = deepcopy(order1)
            sortorder1new = deepcopy(order1new)
            sortorder1newiter = deepcopy(order1newiter)
            sortorder1newiter_change = deepcopy(order1newiter_change)

            for i in sortorder1:
                (i.sort())
            for i in sortorder1new:
                (i.sort())
            for i in sortorder1newiter:
                (i.sort())
            for i in sortorder1newiter_change:
                (i.sort())

            assert (sortorder1 == sortorder1new == sortorder1newiter == sortorder1newiter_change)
        

            sortorder2 = deepcopy(order2)
            sortorder2new = deepcopy(order2new)
            sortorder2newiter = deepcopy(order2newiter)
            sortorder2newiter_change = deepcopy(order2newiter_change)

            for i in sortorder2:
                (i.sort())
            for i in sortorder2new:
                (i.sort())
            for i in sortorder2newiter:
                (i.sort())
            for i in sortorder2newiter_change:
                (i.sort())

            assert (sortorder2 == sortorder2new == sortorder2newiter == sortorder2newiter_change)


            subgnodes1 = []
            for i in subg1:
                for j in i:
                    subgnodes1.append(j)
            subgnodes1 = set(subgnodes1)

            subgnodes2 = []
            for i in subg2:
                for j in i:
                    subgnodes2.append(j)
            subgnodes2 = set(subgnodes2)
            order1_flat = [el for region in order1 for el in region]
            order1_flat = set(order1_flat)

            assert (len(subgnodes1) == len(order1_flat))

            
            order2_flat = [el for region in order2 for el in region]
            order2_flat = set(order2_flat)
            assert (len(subgnodes2) == len(order2_flat))






