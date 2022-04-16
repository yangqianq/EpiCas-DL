'''
-The sequence and epigenetic feature processing code was mainly modified from Doench, John G et al.doi:10.1038/nbt.3026
-The epigenetic annoted code mainly modified from https://github.com/mhorlbeck/CRISPRiaDesign
'''

import numpy as np
import pandas as pd
import sys
import pybedtools
from sklearn import preprocessing as pp
import os

#path = "/srv/shiny-server/EpiCasAPP/APP/py"


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
    rna = pd.read_csv(path + '/epi_reference/%s_rnaseq.csv'%(cell_line), index_col = 3)
    if gene in rna.index:
        rnaRow = rna.loc[gene]
        RNA = rnaRow['RNA']
    else:
        RNA = 0
    return RNA
