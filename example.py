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
    parser.add_argument("-seq", default=None, type=str, help="the input sequence") 
    parser.add_argument("-model", default=None, type=str, help="the path of model path")
    parser.add_argument("-chrom", default=None, type=str, help="the chrom of input sequence")
    parser.add_argument("-start", default=None, type=str, help="the start of input sequence")
    parser.add_argument("-end", default=None, type=str, help="the end of input sequence")
    parser.add_argument("-pamCoord", default=None, type=str, help="the position of pam")
    parser.add_argument("-transcript", default=False, type=str, help="the transcript of inputsequece, please refer our article discription")
    parser.add_argument("-gene", default=None, type=str, help="the gene of input sequence")    
    parser.add_argument("-cell_line", default='Hek293t', type=int, help="The cell line you need")
    parser.add_argument("-end", default=None, type=int, help="proportion end")
    #parser.add_argument("-batch", default=None, type=str, help="path to the batch file")
    args = parser.parse_args()

    return args 

def main_pip(sgRNAseq,model_path, chrom=None, start = None, end =None, pamCoord = None, transcript = None, gene=None, cell_line=None):
    print("checking sequence input...")
    onehot, bplength = seq(sgRNAseq)
    
    chrom = [chrom]
    start = [start]
    end = [end]
    pos = pd.DataFrame(columns = ['chrom','start','end'])
    pos['chrom'],pos['start'],pos['end'] = chrom, start, end
    pos.to_csv(os.path.join('/home/shiny/tempfile','%s_pos.bed'%(cell_line)),sep = '\t',index = False, header = False)
    
    class_threholds = {'23_seq':'CRISPRa_seq','23_epi':'CRISPRa_epi','40_seq':'CRISPRoff_seq','40_epi':'CRISPRoff_epi'}
    
    if 'seq' in model_path:
            inputs = onehot
            class_threhold = class_threholds[str(bplength)+"_seq"]
            efficiency = predict(inputs, model_path, class_threhold)
    else:
        epi_data = endo_annotation(chrom = chrom,start= start, end = end, pamCoord = pamCoord, transcript=transcript, gene=gene, cell_line=cell_line, bplength = bplength)
#         endo_annotation(chrom,start, end, pamCoord, transcript, gene, cell_line,bplength)
        class_threhold = class_threholds[str(bplength)+"_epi"]
        inputs = np.concatenate((onehot, epi_data), axis=3)
        
        efficiency = [predict(inputs, model_path, class_threhold)]
    return efficiency


def seq(sequence):#sequence check and one-hot encoding
    bplength = len(sequence)
    if bplength == 40:
        if sequence[30:32] != 'GG':
            print("Input sequence with incorrect form! correct format:10bp+20bpsgRNA+3bpPAM+7bp")
            sys.exit(1)
    elif bplength == 23:
        if sequence[21:23] != 'GG':
            print("Input sequence with incorrect form! correct format: 20bpsgRNA+3bpPAM")
            sys.exit(1)
    data = pd.DataFrame(columns=(['sequence']))
    data['sequence'] = [sequence]
    x_data = data.iloc[:, 0]
    onehontseq = grna_preprocess(x_data,bplength)
    
    return onehontseq,bplength#1*1*bplength*4


def endo_annotation(chrom,start, end, pamCoord, transcript, gene, cell_line,bplength):
    tss = tss_annoted(gene=gene, pamCoord=pamCoord, transcript=transcript)
    atac = atac_annoted(chrom=chrom, start=start, end=end, cell_line=cell_line)
    methylation = methylation_annoted(chrom=chrom, start=start, end=end, cell_line=cell_line)
    rna = RNA_annoted(gene=gene, cell_line=cell_line)
    
    epi_data = tss
    epi_data['atac'] = [atac]
    epi_data['methylation'] = [methylation]
    epi_data['rna'] = [rna]
    Epi_data = epi_data.fillna(0)
    
    min_max = {'min_tss':-2500, 'max_tss':2500, 'min_atac':0, 'max_atacoff':348.5, 'max_ataca': 2080, 'min_methylation':0, 'max_methylation':100, 'min_rna':0, 'max_rnaoff':822.8642807, 'max_rnaa':19609.4345}
    min_tss, max_tss, min_atac, min_methylation, max_methylation, min_rna = min_max['min_tss'], min_max['max_tss'],min_max['min_atac'],min_max['min_methylation'],min_max['max_methylation'],min_max['min_rna']
    if bplength == 23:
        max_atac = min_max['max_ataca']
        max_rna = min_max['max_rnaa']
    else:
        max_atac = min_max['max_atacoff']
        max_rna = min_max['max_rnaoff']
           
    Epi_data['primary TSS-Up1'] = [(tss['primary TSS-Up'] - min_tss)/(max_tss - min_tss)][0]
    Epi_data['primary TSS-Down1'] = [(tss['primary TSS-Down'] - min_tss)/(max_tss - min_tss)][0]
    Epi_data['secondary TSS-Up1'] = [(tss['secondary TSS-Up'] - min_tss)/(max_tss - min_tss)][0]
    Epi_data['secondary TSS-Down1'] = [(tss['secondary TSS-Down'] - min_tss)/(max_tss - min_tss)][0]
    Epi_data['atac1'] = [(Epi_data['atac'] - min_atac)/(max_atac - min_atac)][0]
    Epi_data['methylation1'] = [(Epi_data['methylation'] - min_methylation)/(max_methylation - min_methylation)][0]
    Epi_data['rna1'] = [(Epi_data['rna'] - min_rna)/(max_rna - min_rna)][0]

    tss1 = Epi_data['primary TSS-Up1']
    tss1 = epi_progress(tss1,bplength)
    tss2 = Epi_data['primary TSS-Down1']
    tss2 = epi_progress(tss2,bplength)
    tss3 = Epi_data['secondary TSS-Up1']
    tss3 = epi_progress(tss3,bplength)
    tss4 = Epi_data['secondary TSS-Down1']
    tss4 = epi_progress(tss4,bplength)
    atac = Epi_data['atac1']
    atac = epi_progress(atac,bplength)
    methylation = Epi_data['methylation1']
    methylation = epi_progress(methylation,bplength)
    rna = Epi_data['rna1']  
    rna = epi_progress(rna,bplength)
    epi_data = np.concatenate((tss1, tss2, tss3, tss4, atac, methylation, rna), axis=3)
    return epi_data


# model_type = "CRISRPoff_epi"
def predict(inputs, model_path, class_threhold):
    
    #models = tf.keras.models.load_model(os.path.join(os.getcwd(),'models',model_type+'s.h5'))
    #modelc = tf.keras.models.load_model(os.path.join(os.getcwd(),'models',model_type+'c.h5'))
    models = tf.keras.models.load_model(os.path.join(model_path+'s.h5'))
    modelc = tf.keras.models.load_model(os.path.join(model_path+'c.h5'))
    
    efficiency_s = models.predict(inputs)[0][0]
    efficiency_c = modelc.predict(inputs)[0][0]
    class_threholds = {'CRISPRa_seq':0.011863946, 'CRISPRa_epi':0.011333015, 'CRISPRoff_seq': 0.2838078, 'CRISPRoff_epi':0.35562915}
    class_threhold = class_threholds[class_threhold]
    if efficiency_c>=class_threhold:
        label = 1
    else:
        label = 0
    return efficiency_s, efficiency_c, class_threhold, label

if __name__ == "__main__":
    main_pip()
