import numpy as np
import pandas as pd
import tensorflow as tf
import configparser
import os
import argparse
import pickle
from function import*
import pybedtools
from pybedtools import BedTool

def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sequence", default=None, type=str, help="the input sequence") 
    parser.add_argument("-model", default=None, type=str, help="the path of model path")
    parser.add_argument("-chrom", default=None, type=str, help="the chrom of input sequence")
    parser.add_argument("-start", default=None, type=str, help="the start of input sequence")
    parser.add_argument("-end", default=None, type=str, help="the end of input sequence")
    parser.add_argument("-pamCoord", default=None, type=str, help="the position of pam")
    parser.add_argument("-transcript", default=False, type=str, help="the transcript of inputsequece, please refer our article discription")
    parser.add_argument("-gene", default=None, type=str, help="the gene of input sequence")    
    parser.add_argument("-cell_line", default='Hek293t', type=str, help="The cell line you need")
    #parser.add_argument("-batch", default=None, type=str, help="path to the batch file")
    args = parser.parse_args()

    return args 
    

def main_pip():
    args = Args()
    print("checking sequence input...")
    onehot, bplength = seq(args.sequence)
     
    chrom = [args.chrom]
    start = [args.start]
    end = [args.end]
    pos = pd.DataFrame(columns = ['chrom','start','end'])
    pos['chrom'],pos['start'],pos['end'] = chrom, start, end
    pos.to_csv('./tempfile/%s_pos.bed'%(args.cell_line),sep = '\t',index = False, header = False)
    
    class_threholds = {'23_seq':'CRISPRa_seq','23_epi':'CRISPRa_epi','40_seq':'CRISPRoff_seq','40_epi':'CRISPRoff_epi'}
    
    if 'seq' in args.model:
            inputs = onehot
            class_threhold = class_threholds[str(bplength)+"_seq"]
            efficiency = predict(inputs, args.model, class_threhold)
    else:
        epi_data = endo_annotation(chrom = chrom,start= start, end = end, pamCoord = args.pamCoord, transcript=args.transcript, gene=args.gene, cell_line=args.cell_line, bplength = bplength)
#         endo_annotation(chrom,start, end, pamCoord, transcript, gene, cell_line,bplength)
        class_threhold = class_threholds[str(bplength)+"_epi"]
        inputs = np.concatenate((onehot, epi_data), axis=3)
        
        efficiency = [predict(inputs, args.model, class_threhold)]
    
    shutil.rmtree('./tempfile')  
    os.mkdir('./tempfile')
    print("Predicted Phenotype Score:"+str(efficiency[0]),"Probabilities:"+str(efficiency[1]),"Threshold:"+str(efficiency[2]),"classification:"+str(efficiency[3]))

if __name__ == "__main__":
    main_pip()