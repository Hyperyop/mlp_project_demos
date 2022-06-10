import numpy as np
from pathlib import Path
def parser(batches,ll,el,vectorize_layer,dic_length,batch_size=79): 
    """
    takes an molecule batches represented by i and process it.
    input:
     - batches : molecule batches in the form [batches , (node_type, edges, node_target)]
     - ll : maximum nodes
     - el : maximum edges
     - vectorize_layer : transforms the atoms from letters to integers
     - dic_length : ?
     - batch_size : batch_size to be used 
    outputs:
     - ai:nodes representation
     - ei : inbound edges
     - eo : outbound edges 
     - ao : nodes targets
    """
    a,b,d=zip(*batches)
    bl=np.array([len(x) for x in a])
    bel=np.array([len(x[0]) for x in b])
    eligible=np.logical_and(bl<ll, bel<el)


    ao=np.zeros((eligible.sum(), ll))
    ai=np.zeros((eligible.sum(), ll))
    ei=np.zeros((eligible.sum(), el),dtype=np.int32)
    eo=np.zeros((eligible.sum(), el),dtype=np.int32)

    eligible=np.where(eligible)[0]
    for i in range(len(eligible)):
        j=eligible[i]
        ai[i]=np.pad(vectorize_layer(a[j])[:,0], pad_width=(0, ll-len(a[j])), constant_values=(dic_length, dic_length))
        ei[i]=np.pad(b[j][0], pad_width=(0, el-len(b[j][0])), constant_values=(ll-1, ll-1))
        eo[i]=np.pad(b[j][1], pad_width=(0, el-len(b[j][0])), constant_values=(ll-1, ll-1))
        ao[i]=np.pad(d[j], pad_width=(0, ll-len(a[j])), constant_values=(0, 0))
#     mask=ao!=0
    mask=ai<dic_length
    return ao,ai,ei,eo,mask
def create_batches(input_path : Path,num=100):
    """
    Creates batches from the npz files.
    input : 
     - input_path: a Path instance 
     - num : number of instances to load
    output : 
     - a list of batches
    """
    number_of_iterations=0
    batches= []
    for i in input_path.glob("*.npz"):
        number_of_iterations+=1
        if (number_of_iterations>num):
            break

        with np.load(i) as tempfile:
            tempfile_names = tempfile.files
            batches.append([tempfile[tempfile_names[1]],tempfile[tempfile_names[-1]],tempfile[tempfile_names[2]]])
    return batches

if __name__=="__main__":
    print("please use this as module")