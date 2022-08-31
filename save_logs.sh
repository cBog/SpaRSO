#!/bin/bash

dir_array=(
    logs/20220831_1736_da00fbe4e2
    logs/20220829_1935_faa2d8b2ba    
    logs/20220829_1952_faa2d8b2ba    
    logs/20220826_1400_c6ab1f8864
    logs/20220826_1621_a634e5b000
    logs/20220827_1048_45b9eb8795
    logs/20220827_2120_dc1ed85beb
    logs/20220828_0606_dc1ed85beb
    logs/20220828_1309_ad8d4348f8
    logs/20220828_1833_ad8d4348f8
    logs/20220829_0044_9c5aebed3c
    logs/20220828_2020_82bf403a2f
    logs/20220828_2240_79afd14a6b
    logs/20220829_0101_9c5aebed3c
    logs/20220829_0340_9c5aebed3c
    logs/20220829_0606_9c5aebed3c
    logs/20220828_1215_ad8d4348f8
    logs/20220828_1456_ad8d4348f8
    logs/20220828_1834_ad8d4348f8
    logs/20220828_2316_79afd14a6b
    logs/20220829_0224_9c5aebed3c
    logs/20220828_2048_118aace4a3
    logs/20220829_0046_9c5aebed3c
    logs/20220829_0431_9c5aebed3c
    logs/20220829_0552_9c5aebed3c
    logs/20220829_1301_0891ac381b
    logs/20220828_2103_79afd14a6b
    logs/20220828_2340_79afd14a6b
    logs/20220829_0223_9c5aebed3c
    logs/20220829_1318_1f4d3bea6b
    logs/20220829_1323_1f4d3bea6b
    logs/20220830_1103_95fdf2e55d
    logs/20220831_0953_e15020ca50
    logs/20220831_1735_da00fbe4e2
    logs/Exp1
    logs/Exp2
    logs/Exp3
    logs/Exp4_1
    logs/Exp4_2
    logs/Exp5
    logs/Exp6
    logs/FinalRuns
    logs/ReportPlots
)

for d in "${dir_array[@]}";do
 echo $d
 cp -r $d saved_logs/${d}
done