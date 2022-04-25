'''
-The sequence and epigenetic feature processing code was mainly modified from Doench, John G et al.doi:10.1038/nbt.3026
-The epigenetic annoted code mainly modified from https://github.com/mhorlbeck/CRISPRiaDesign
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import pybedtools
from pybedtools import BedTool
from sklearn import preprocessing as pp
import os

path = os.getcwd()
##比如当前的路径就是/OceanStor100D/home/sunyidi_lab/qqyang/sgRNAtool/other_model/some_test/3_26_test/github_code



def grna_preprocess(lines,length):
    #length = 40
    data_n = len(lines)
    seq = np.zeros((data_n, 1, length, 4), dtype=int)
    for l in range(data_n):
        data = lines[l]
        seq_temp = data
        for i in range(length):
            if seq_temp[i] in "Aa":
                seq[l, 0, i, 0] = 1
            elif seq_temp[i] in "Cc":
                seq[l, 0, i, 1] = 1
            elif seq_temp[i] in "Gg":
                seq[l, 0, i, 2] = 1
            elif seq_temp[i] in "Tt":
                seq[l, 0, i, 3] = 1
    return seq    
    


def epi_progress(lines, length = 40):
    data_n = len(lines)
    epi = np.zeros((data_n, 1, length, 1), dtype=float)
    for l in range(data_n):
        for i in range(length):
            epi[l, 0, i, 0] = lines[l]
    return epi


# p1p2Table.head()##中间两行的作用应该是对数据框中的数据类型进行转换
# models = tf.keras.models.load_model(os.path.join(os.getcwd(),'models',model_type+'s.h5'))

def tss_annoted(gene, pamCoord, transcript = False):
    p1p2Table = pd.read_csv('./epi_reference/human_p1p2Table.txt',sep='\t', header=0, index_col=[0,1])
#     p1p2Table = pd.read_csv('./epi_reference/human_p1p2Table.txt',sep='\t', header=0, index_col=[0,1])#有两个列数据均为index
    p1p2Table['primary TSS'] = p1p2Table['primary TSS'].apply(lambda tupString: (int(tupString.strip('()').split(', ')[0].split('.')[0]), int(tupString.strip('()').split(', ')[1].split('.')[0])))
    p1p2Table['secondary TSS'] = p1p2Table['secondary TSS'].apply(lambda tupString: (int(tupString.strip('()').split(', ')[0].split('.')[0]),int(tupString.strip('()').split(', ')[1].split('.')[0])))
    sgDistanceSeries = []
    pamCoord = int(pamCoord)
    T_list = ['P1', 'P2', 'P1P2']
    if transcript in T_list:
        Transcript = True
    else:
        Transcript = False
        
    if Transcript == False:
        if gene in p1p2Table.index:
            tssRow = p1p2Table.loc[gene]
            if len(tssRow) ==1:
                tssRow = tssRow.iloc[0]
                if tssRow['strand'] =='+':
                    sgDistanceSeries.append((pamCoord - tssRow['primary TSS'][0],
                                            pamCoord - tssRow['primary TSS'][1],
                                            pamCoord - tssRow['secondary TSS'][0],
                                            pamCoord - tssRow['secondary TSS'][1]))
                else:
                    sgDistanceSeries.append(((pamCoord - tssRow['primary TSS'][1]) * -1,
                                            (pamCoord - tssRow['primary TSS'][0]) * -1,
                                            (pamCoord - tssRow['secondary TSS'][1]) * -1,
                                            (pamCoord - tssRow['secondary TSS'][0]) * -1))
            else:
                closestTssRow = tssRow.loc[tssRow.apply(lambda row: abs(pamCoord - row['primary TSS'][0]), axis=1).idxmin()]
                if closestTssRow['strand'] == '+':
                    sgDistanceSeries.append((pamCoord - closestTssRow['primary TSS'][0],
                                            pamCoord - closestTssRow['primary TSS'][1],
                                            pamCoord - closestTssRow['secondary TSS'][0],
                                            pamCoord - closestTssRow['secondary TSS'][1]))
                else:
                    sgDistanceSeries.append(((pamCoord - closestTssRow['primary TSS'][1]) * -1,
                                            (pamCoord - closestTssRow['primary TSS'][0]) * -1,
                                            (pamCoord - closestTssRow['secondary TSS'][1]) * -1,
                                            (pamCoord - closestTssRow['secondary TSS'][0]) * -1))
    else:
        name = (gene, transcript)
        if name in p1p2Table.index:
            tssRow = p1p2Table.loc[[name]]
            if len(tssRow) == 1:
                tssRow = tssRow.iloc[0]
#                 print(tssRow)
                if tssRow['strand'] == '+':
                    sgDistanceSeries.append((pamCoord - tssRow['primary TSS'][0],
                                            pamCoord - tssRow['primary TSS'][1],
                                            pamCoord - tssRow['secondary TSS'][0],
                                            pamCoord - tssRow['secondary TSS'][1]))
                else:
                    sgDistanceSeries.append(((pamCoord - tssRow['primary TSS'][1]) * -1,
                                            (pamCoord - tssRow['primary TSS'][0]) * -1,
                                            (pamCoord - tssRow['secondary TSS'][1]) * -1,
                                            (pamCoord - tssRow['secondary TSS'][0]) * -1))
            else:
                print(name, tssRow)
                raise ValueError('all gene/trans paris should be unique')
    return pd.DataFrame(sgDistanceSeries, columns=['primary TSS-Up', 'primary TSS-Down', 'secondary TSS-Up', 'secondary TSS-Down'])



def atac_annoted(chrom, start, end, cell_line = 'Hek293t'):
    atac_bed = pybedtools.BedTool('./epi_reference/%s_atac.bed'%(cell_line))
    pos = pybedtools.BedTool('./tempfile/%s_pos.bed'%(cell_line))
    pos_atac = pos.intersect(atac_bed, wb = True)
    pos_atac.moveto('./tempfile/%s_posatac.bed'%(cell_line))
    pos_atac = pd.read_csv('./tempfile/%s_posatac.bed'%(cell_line), sep='\t',names = ["chrom","start","end","chrom_1","statr_1","end_1","atac"])
    if len(pos_atac) == 1:
        print(pos_atac['atac'][0])
    elif len(pos_atac) == 0:
        print(0)
    else:
        max_atac = pos_atac['atac'].argmax()##series 找最大值 .idxmax()用于返回数据框中的最大值，这个返回的是一个序列位置
        print(pos_atac['atac'][max_atac])
        
        
def methylation_annoted(chrom,start,end,cell_line = 'Hek293t'):
    methylation_bed = pybedtools.BedTool('./epi_reference/%s_methylation.bed'%(cell_line))
    pos = pybedtools.BedTool('./tempfile/%s_pos.bed'%(cell_line))
    pos_methylation = pos.intersect(methylation_bed, wb = True)
    pos_methylation.moveto('./tempfile/%s_posmethylation.bed'%(cell_line))
    pos_methylation = pd.read_csv('./tempfile/%s_posmethylation.bed'%(cell_line),sep = '\t',names = ["chrom","start","end","chrom_1","statr_1","end_1","methylation"])
    
    if len(pos_methylation) == 1:
        print(pos_methylation['methylation'][0])
    elif len(pos_methylation) == 0:
        print(0)
    else:
        max_methylation = pos_methylation['methylation'].argmax()##series 找最大值 .idxmax()用于返回数据框中的最大值
        print(pos_methylation['methylation'][max_methylation])
        
        
        
def RNA_annoted(gene, cell_line = 'Hek293t'):
#     rna = pd.read_csv('./epi_reference/%s_rnaseq.csv'%(cell_line), index_col = 3)
    rna = pd.read_csv('./epi_reference/%s_rnaseq.csv'%(cell_line), index_col = 3)
    if gene in rna.index:
        rnaRow = rna.loc[gene]
        RNA = rnaRow['RNA']
    else:
        RNA = 0
    return RNA


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
def predict(inputs, model, class_threhold):
    
    #models = tf.keras.models.load_model(os.path.join(os.getcwd(),'models',model_type+'s.h5'))
    #modelc = tf.keras.models.load_model(os.path.join(os.getcwd(),'models',model_type+'c.h5'))
    models = tf.keras.models.load_model(os.path.join('./models/'+model+'s.h5'))
    modelc = tf.keras.models.load_model(os.path.join('./models/'+model+'c.h5'))
#     modelc = tf.keras.models.load_model(os.path.join(model_path+'c.h5'))
    
    efficiency_s = models.predict(inputs)[0][0]
    efficiency_c = modelc.predict(inputs)[0][0]
    class_threholds = {'CRISPRa_seq':0.011863946, 'CRISPRa_epi':0.011333015, 'CRISPRoff_seq': 0.2838078, 'CRISPRoff_epi':0.35562915}
    class_threhold = class_threholds[class_threhold]
    if efficiency_c>=class_threhold:
        label = 1
    else:
        label = 0
    return efficiency_s, efficiency_c, class_threhold, label

