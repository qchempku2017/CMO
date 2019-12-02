from gurobi import *
from copy import deepcopy

"""
    Auxiliary networks encoding.
"""


def add_softcls_terms(m,x,bcs,ecis):
    """
    Builds an auxilary network to reformulate the problem into binary quadratic programming.
    m: Gurobi model
    x: binary bits
    bcs: bit clusters
    ecis: eci of bit clusters
    """
    all_vars = [var for var in x]
    aux_id0 = len(all_vars) #id of the first aux var in all_vars
    all_v_pairs = [] #All axiliary variables, expressed by their original pair of variables. v_aux = (v_ori[1],v_ori[2])

    obj = GenExpr()
    for bc,eci in zip(bcs,ecis):
        if len(bc)<=2:
        #No need for reduction
            clusterm = eci
            for bit in bc:
                clusterm = clusterm*x[bit-1]
            obj+=clusterm
        else:
        #Add auxilary variables and reduce.
            current_bc = deepcopy(bc)
            
            current_bc = pre_process_bc(current_bc,all_v_pairs,aux_id0)
            #Only pre process once, because any new added variable can not appear in the existing set.
            while len(current_bc)>2:
                reduced_bc = []

                #divide the cluster into batches with length 2.
                batches = []
                for ba_id in range(len(current_bc)//2):
                    batches.append(current_bc[2*ba_id:2*(ba_id+1)])
                if len(current_bc)%2 == 1:
                    batches.append([current_bc[-1]])
                
                for batch in batches:
                    if len(batch)>1: 
                        #If so, then this batch is a new pair that hasn't been auxiliarized before.
                        aux_v_ba = m.addVar(vtype=GRB.BINARY)
                        all_vars.append(aux_v_ba)
                        all_v_pairs.append(batch)
                        m.addConstr(all_vars[batch[0]-1]*all_vars[batch[1]-1] == all_vars[len(all_vars)-1])

                        reduced_bc.append(len(all_vars)) 
                   #index of the current last auxiliary variable (the newly added one) in all_vars. For some historical reason, the index is 1-based, not 0-based.
                    else:
                        reduced_bc.append(batch[0])

                current_bc = reduced_bc
            
            clusterm = eci
            for bit in current_bc:
                clusterm = clusterm*all_vars[bit-1]
            obj +=clusterm

    m.setObjective(obj,GRB.MINIMIZE)
    m.update()

def pre_process_bc(current_bc,all_v_pairs,aux_id0):
    """
    Preprocess a cluster to remove already auxilarized pairs in it, so we can minimize the number of auxiliary vars
    to use.
    """
    processed_bc = deepcopy(current_bc)
    for p_id,v_pair in enumerate(all_v_pairs):
        if v_pair[0] in processed_bc and v_pair[1] in processed_bc:
            processed_bc.remove(v_pair[0])
            processed_bc.remove(v_pair[1])
            processed_bc.append(p_id+aux_id0+1)
            #replace this pair with existing aux variables in advance.
    return processed_bc


