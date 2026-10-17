# Fase D - walk-forward mensal (2026-10-16)

## Amplitude (teste 2014-10-01 a 2026-10-16)

                     mae_pp                               ganho_mae_vs_har_22d (baseline)_%                               ganho_mae_vs_har_5d (baseline)_%                               spearman                              
periodo            completo ultimos_2_anos sem_05_10_2026                          completo ultimos_2_anos sem_05_10_2026                         completo ultimos_2_anos sem_05_10_2026 completo ultimos_2_anos sem_05_10_2026
modelo                                                                                                                                                                                                                         
har_5d (baseline)    0.5123         0.4539         0.5124                            3.4800        -4.0730         3.4624                           0.0000         0.0000         0.0000   0.5471         0.3032         0.5469
har_22d (baseline)   0.5308         0.4362         0.5308                            0.0000         0.0000         0.0000                          -3.6055         3.9136        -3.5866   0.5337         0.2917         0.5338
media_historica      0.6664         0.6341         0.6665                          -25.5561       -45.3717       -25.5675                         -30.0830       -39.6824       -30.0711   0.2742        -0.2471         0.2751
HAR_ols_log          0.4899         0.4229         0.4900                            7.6966         3.0387         7.6889                           4.3686         6.8334         4.3781   0.5701         0.3217         0.5699
HAR+VIX_ols_log      0.4870         0.4242         0.4870                            8.2570         2.7383         8.2532                           4.9492         6.5447         4.9626   0.5779         0.3235         0.5778
ridge_todas          0.5116         0.4575         0.5118                            3.6090        -4.8921         3.5815                           0.1337        -0.7870         0.1234   0.5658         0.3012         0.5656
hgb_todas            0.5276         0.4388         0.5277                            0.5962        -0.5969         0.5729                          -2.9879         3.3401        -2.9931   0.5662         0.3362         0.5660
rf_todas             0.5233         0.4425         0.5235                            1.4048        -1.4494         1.3687                          -2.1501         2.5210        -2.1688   0.5717         0.3317         0.5714

## Lateral (teste 2014-10-01 a 2026-10-16)

                             brier                               ganho_brier_vs_taxa_252d (baseline)_%                                    auc                               taxa_real_top20%                              
periodo                   completo ultimos_2_anos sem_05_10_2026                              completo ultimos_2_anos sem_05_10_2026 completo ultimos_2_anos sem_05_10_2026         completo ultimos_2_anos sem_05_10_2026
modelo                                                                                                                                                                                                                    
taxa_252d (baseline)        0.2131         0.1756         0.2132                                0.0000         0.0000         0.0000   0.5867         0.5100         0.5866           0.8004         0.7573         0.8004
logit_HAR+VIX | expansiva   0.2063         0.1774         0.2063                                3.2016        -1.0370         3.2118   0.6504         0.5246         0.6504           0.8137         0.8000         0.8137
logit_HAR+VIX | movel_5a    0.2061         0.1759         0.2061                                3.3161        -0.1822         3.3178   0.6469         0.5276         0.6468           0.8118         0.8000         0.8118
logit_todas | expansiva     0.2063         0.1779         0.2063                                3.2127        -1.3449         3.2209   0.6495         0.5238         0.6495           0.8176         0.7800         0.8176
logit_todas | movel_5a      0.2062         0.1772         0.2063                                3.2549        -0.9023         3.2546   0.6446         0.5137         0.6445           0.8176         0.7800         0.8176
hgb_todas | expansiva       0.2088         0.1823         0.2088                                2.0431        -3.8412         2.0612   0.6297         0.5155         0.6298           0.8176         0.7700         0.8176
hgb_todas | movel_5a        0.2126         0.1809         0.2126                                0.2562        -3.0217         0.2645   0.6201         0.5188         0.6201           0.8000         0.8000         0.8000
taxa_media | expansiva      0.2188         0.1883         0.2189                               -2.6742        -7.2519        -2.6650   0.5616         0.4599         0.5614           0.7830         0.7521         0.7826
taxa_media | movel_5a       0.2170         0.1797         0.2170                               -1.8039        -2.3635        -1.8025   0.5575         0.4665         0.5572           0.7830         0.7672         0.7826

### Calibracao por quintil (logit_HAR+VIX)

                                  prob_media                               taxa_real                              
periodo                             completo ultimos_2_anos sem_05_10_2026  completo ultimos_2_anos sem_05_10_2026
modelo                    quintil                                                                                 
logit_HAR+VIX | expansiva 1            0.506          0.642          0.506     0.498          0.760          0.498
                          2            0.617          0.709          0.617     0.629          0.747          0.631
                          3            0.663          0.733          0.663     0.694          0.808          0.692
                          4            0.706          0.751          0.706     0.767          0.768          0.766
                          5            0.752          0.771          0.752     0.814          0.800          0.814
logit_HAR+VIX | movel_5a  1            0.492          0.647          0.492     0.510          0.760          0.510
                          2            0.614          0.739          0.614     0.624          0.768          0.623
                          3            0.664          0.767          0.664     0.686          0.778          0.686
                          4            0.717          0.791          0.717     0.771          0.778          0.770
                          5            0.780          0.822          0.780     0.812          0.800          0.812

## Gap (teste 2025-10-01 a 2026-10-16)

                        mae_pp                               ganho_mae_vs_zero_%                                 r2_oos                               acerto_direcao                              
periodo               completo ultimos_2_anos sem_05_10_2026            completo ultimos_2_anos sem_05_10_2026 completo ultimos_2_anos sem_05_10_2026       completo ultimos_2_anos sem_05_10_2026
modelo                                                                                                                                                                                            
zero (baseline)         0.5487         0.5487         0.5231              0.0000         0.0000         0.0000   0.0000         0.0000         0.0000          0.000          0.000         0.0000
ridge_EWZ               0.4775         0.4775         0.4702             12.9801        12.9801        10.1195   0.4180         0.4180         0.2334          0.688          0.688         0.6867
ridge_EWZ+ES            0.4802         0.4802         0.4690             12.4861        12.4861        10.3448   0.3968         0.3968         0.2479          0.672          0.672         0.6707
ridge_EWZ+VIX           0.4802         0.4802         0.4696             12.4842        12.4842        10.2243   0.4036         0.4036         0.2493          0.664          0.664         0.6627
ridge_EWZ+BRENT         0.4905         0.4905         0.4826             10.6024        10.6024         7.7471   0.3847         0.3847         0.1937          0.624          0.624         0.6225
ridge_todas             0.4939         0.4939         0.4822              9.9859         9.9859         7.8316   0.3677         0.3677         0.2159          0.672          0.672         0.6707
ridge_todas_sem_EWZ     0.5398         0.5398         0.5140              1.6230         1.6230         1.7532   0.0614         0.0614         0.0919          0.620          0.620         0.6225
ridge_todas_sem_ES      0.4939         0.4939         0.4832              9.9740         9.9740         7.6290   0.3725         0.3725         0.2092          0.660          0.660         0.6586
ridge_todas_sem_VIX     0.4929         0.4929         0.4815             10.1642        10.1642         7.9545   0.3704         0.3704         0.2149          0.672          0.672         0.6707
ridge_todas_sem_BRENT   0.4816         0.4816         0.4700             12.2246        12.2246        10.1489   0.3937         0.3937         0.2491          0.676          0.676         0.6747

## Retorno do dia (referencia)

                               mae_pp                               ganho_mae_vs_zero_%                               acerto_direcao                              
periodo                      completo ultimos_2_anos sem_05_10_2026            completo ultimos_2_anos sem_05_10_2026       completo ultimos_2_anos sem_05_10_2026
alvo         modelo                                                                                                                                               
ret_cc       zero (baseline)   0.8933         0.8933         0.8676              0.0000         0.0000         0.0000         0.0000         0.0000         0.0000
             ridge_todas       0.8625         0.8625         0.8506              3.4524         3.4524         1.9577         0.5444         0.5444         0.5426
             ridge_EWZ         0.8564         0.8564         0.8457              4.1341         4.1341         2.5332         0.5560         0.5560         0.5543
ret_intraday zero (baseline)   0.7086         0.7086         0.7099              0.0000         0.0000         0.0000         0.0000         0.0000         0.0000
             ridge_todas       0.7142         0.7142         0.7155             -0.7887        -0.7887        -0.7892         0.4885         0.4885         0.4903
             ridge_EWZ         0.7131         0.7131         0.7110             -0.6344        -0.6344        -0.1547         0.5308         0.5308         0.5328

## Vencedores (models/)

- amplitude: HAR+VIX_ols_log (expansiva); vence tambem em: {'completo': True, 'ultimos_2_anos': False, 'sem_05_10_2026': True}; treino 2009-06-03 a 2026-10-16 (n = 3399)
- lateral: logit_HAR+VIX | movel_5a (movel 5 anos); vence tambem em: {'completo': True, 'ultimos_2_anos': False, 'sem_05_10_2026': True}; treino 2021-10-18 a 2026-10-16 (n = 1248)
- gap: ridge_EWZ (expansiva); vence tambem em: {'completo': True, 'ultimos_2_anos': True, 'sem_05_10_2026': False}; treino 2024-10-24 a 2026-10-16 (n = 492)
