EpiCas-DL
===
EpiCas-DL is a deeplearning based prediction tool for the sgRNA editing effects of CRISPR/dCas editing system.The tool combined four types of epigenetic features, including methylation, gene expression, chromatin accessibility and the distance between target site and transcription start site, to improve the prediction of our tool.

System Requirements:
---
    Red Hat 4.8.5-36
    python 3.7
    python packages:
         tensorflow 2.4.1
         numpy 1.20.1
         pandas 1.2.4
         pybedtools 0.8.2
         GPyOpt 1.2.6

Installation
---    
    
      git clone https://github.com/yangqianq/EpiCas-DL.git
      
Useage
---
      # Predicted only for sequence
      python example.py -sequence [input-sequence] -model [path-to-save-seqmodel]
      python example.py -sequence GGGCCCAGATCCCTCTATGTGCTCGAAGCAGGTGGACCCC -model CRISPRoff_seq
      
      # Predicted included both sequence and epigenetic features
      python example.py -sequence [input-sequence] -model [path-to-save-epimodel] -chrom [chromosome-of-input-sequence] -start [start-position-of-input-sequence] -end [end-position-of-input-sequence] -pamCoord [pam-position] -transcription [transcription-of-input-sequence] -gene [gene-of-input-sequence] -cell_line [cell-line-of-input-sequence]
      python example.py -sequence GGGCCCAGATCCCTCTATGTGCTCGAAGCAGGTGGACCCC -model CRISPRoff_epi -chrom chr12 -start 100592057 -end 100592096 -pamCoord 100592087 -transcript P1P2 -gene ACTR6 -cell_line Hek293t
      
      
Note
---

1.There have CRISPRoff_seq/CRISPRoff_epi/CRISPRa_seq/CRISPRa_epi four models for you. 

2.We provide "epi_reference" folder at "https://pan.baidu.com/s/1TbUhdmaJ8b0vhUF1Tqhuqw?pwd=ilg1".

3.We also provided a website that available at http://www.sunlab.fun:3838/EpiCas-DL.

---
If you have any question, please contact us at email yangqq2024@shanghaitech.edu.cn
                
                
