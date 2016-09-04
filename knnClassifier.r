
R version 3.2.2 (2015-08-14) -- "Fire Safety"
Copyright (C) 2015 The R Foundation for Statistical Computing
Platform: x86_64-w64-mingw32/x64 (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

[Previously saved workspace restored]

> wbcd = read.csv("C:\\Users\\NISHANT\\Desktop\\ML\\Machine-Learning-with-R-datasets-master\\wisc_bc_data.csv", stringsAsFactors = FALSE)
> str(wbcd)
'data.frame':   569 obs. of  32 variables:
 $ id                     : int  842302 842517 84300903 84348301 84358402 843786 844359 84458202 844981 84501001 ...
 $ diagnosis              : chr  "M" "M" "M" "M" ...
 $ radius_mean            : num  18 20.6 19.7 11.4 20.3 ...
 $ texture_mean           : num  10.4 17.8 21.2 20.4 14.3 ...
 $ perimeter_mean         : num  122.8 132.9 130 77.6 135.1 ...
 $ area_mean              : num  1001 1326 1203 386 1297 ...
 $ smoothness_mean        : num  0.1184 0.0847 0.1096 0.1425 0.1003 ...
 $ compactness_mean       : num  0.2776 0.0786 0.1599 0.2839 0.1328 ...
 $ concavity_mean         : num  0.3001 0.0869 0.1974 0.2414 0.198 ...
 $ concave.points_mean    : num  0.1471 0.0702 0.1279 0.1052 0.1043 ...
 $ symmetry_mean          : num  0.242 0.181 0.207 0.26 0.181 ...
 $ fractal_dimension_mean : num  0.0787 0.0567 0.06 0.0974 0.0588 ...
 $ radius_se              : num  1.095 0.543 0.746 0.496 0.757 ...
 $ texture_se             : num  0.905 0.734 0.787 1.156 0.781 ...
 $ perimeter_se           : num  8.59 3.4 4.58 3.44 5.44 ...
 $ area_se                : num  153.4 74.1 94 27.2 94.4 ...
 $ smoothness_se          : num  0.0064 0.00522 0.00615 0.00911 0.01149 ...
 $ compactness_se         : num  0.049 0.0131 0.0401 0.0746 0.0246 ...
 $ concavity_se           : num  0.0537 0.0186 0.0383 0.0566 0.0569 ...
 $ concave.points_se      : num  0.0159 0.0134 0.0206 0.0187 0.0188 ...
 $ symmetry_se            : num  0.03 0.0139 0.0225 0.0596 0.0176 ...
 $ fractal_dimension_se   : num  0.00619 0.00353 0.00457 0.00921 0.00511 ...
 $ radius_worst           : num  25.4 25 23.6 14.9 22.5 ...
 $ texture_worst          : num  17.3 23.4 25.5 26.5 16.7 ...
 $ perimeter_worst        : num  184.6 158.8 152.5 98.9 152.2 ...
 $ area_worst             : num  2019 1956 1709 568 1575 ...
 $ smoothness_worst       : num  0.162 0.124 0.144 0.21 0.137 ...
 $ compactness_worst      : num  0.666 0.187 0.424 0.866 0.205 ...
 $ concavity_worst        : num  0.712 0.242 0.45 0.687 0.4 ...
 $ concave.points_worst   : num  0.265 0.186 0.243 0.258 0.163 ...
 $ symmetry_worst         : num  0.46 0.275 0.361 0.664 0.236 ...
 $ fractal_dimension_worst: num  0.1189 0.089 0.0876 0.173 0.0768 ...
> 
> wbcd = wbcd[-1]
> str(wbcd)
'data.frame':   569 obs. of  31 variables:
 $ diagnosis              : chr  "M" "M" "M" "M" ...
 $ radius_mean            : num  18 20.6 19.7 11.4 20.3 ...
 $ texture_mean           : num  10.4 17.8 21.2 20.4 14.3 ...
 $ perimeter_mean         : num  122.8 132.9 130 77.6 135.1 ...
 $ area_mean              : num  1001 1326 1203 386 1297 ...
 $ smoothness_mean        : num  0.1184 0.0847 0.1096 0.1425 0.1003 ...
 $ compactness_mean       : num  0.2776 0.0786 0.1599 0.2839 0.1328 ...
 $ concavity_mean         : num  0.3001 0.0869 0.1974 0.2414 0.198 ...
 $ concave.points_mean    : num  0.1471 0.0702 0.1279 0.1052 0.1043 ...
 $ symmetry_mean          : num  0.242 0.181 0.207 0.26 0.181 ...
 $ fractal_dimension_mean : num  0.0787 0.0567 0.06 0.0974 0.0588 ...
 $ radius_se              : num  1.095 0.543 0.746 0.496 0.757 ...
 $ texture_se             : num  0.905 0.734 0.787 1.156 0.781 ...
 $ perimeter_se           : num  8.59 3.4 4.58 3.44 5.44 ...
 $ area_se                : num  153.4 74.1 94 27.2 94.4 ...
 $ smoothness_se          : num  0.0064 0.00522 0.00615 0.00911 0.01149 ...
 $ compactness_se         : num  0.049 0.0131 0.0401 0.0746 0.0246 ...
 $ concavity_se           : num  0.0537 0.0186 0.0383 0.0566 0.0569 ...
 $ concave.points_se      : num  0.0159 0.0134 0.0206 0.0187 0.0188 ...
 $ symmetry_se            : num  0.03 0.0139 0.0225 0.0596 0.0176 ...
 $ fractal_dimension_se   : num  0.00619 0.00353 0.00457 0.00921 0.00511 ...
 $ radius_worst           : num  25.4 25 23.6 14.9 22.5 ...
 $ texture_worst          : num  17.3 23.4 25.5 26.5 16.7 ...
 $ perimeter_worst        : num  184.6 158.8 152.5 98.9 152.2 ...
 $ area_worst             : num  2019 1956 1709 568 1575 ...
 $ smoothness_worst       : num  0.162 0.124 0.144 0.21 0.137 ...
 $ compactness_worst      : num  0.666 0.187 0.424 0.866 0.205 ...
 $ concavity_worst        : num  0.712 0.242 0.45 0.687 0.4 ...
 $ concave.points_worst   : num  0.265 0.186 0.243 0.258 0.163 ...
 $ symmetry_worst         : num  0.46 0.275 0.361 0.664 0.236 ...
 $ fractal_dimension_worst: num  0.1189 0.089 0.0876 0.173 0.0768 ...
> table(wbcd$diagnosis)

  B   M 
357 212 
> wbcd$diagnosis = factor( wbcd$diagnosis, levels = c("B", "M"), labels = c("Benign" , "Malignant"))
> round(prop.table(table( wbcd$diagnosis)) * 100, digits = 1)

   Benign Malignant 
     62.7      37.3 
> summary( wbcd[c("radius_mean", "area_mean", "smoothness_mean")])
  radius_mean       area_mean      smoothness_mean  
 Min.   : 6.981   Min.   : 143.5   Min.   :0.05263  
 1st Qu.:11.700   1st Qu.: 420.3   1st Qu.:0.08637  
 Median :13.370   Median : 551.1   Median :0.09587  
 Mean   :14.127   Mean   : 654.9   Mean   :0.09636  
 3rd Qu.:15.780   3rd Qu.: 782.7   3rd Qu.:0.10530  
 Max.   :28.110   Max.   :2501.0   Max.   :0.16340  
>  normalize <- function(x) {       return ((x - min(x)) / (max(x) - min(x))) }
> normalize(c(1, 2, 3, 4, 5))
[1] 0.00 0.25 0.50 0.75 1.00
> wbcd_n = as.data.frame( lapply( wbcd[2:31], normalize))
> str( wbcd_n)
'data.frame':   569 obs. of  30 variables:
 $ radius_mean            : num  0.521 0.643 0.601 0.21 0.63 ...
 $ texture_mean           : num  0.0227 0.2726 0.3903 0.3608 0.1566 ...
 $ perimeter_mean         : num  0.546 0.616 0.596 0.234 0.631 ...
 $ area_mean              : num  0.364 0.502 0.449 0.103 0.489 ...
 $ smoothness_mean        : num  0.594 0.29 0.514 0.811 0.43 ...
 $ compactness_mean       : num  0.792 0.182 0.431 0.811 0.348 ...
 $ concavity_mean         : num  0.703 0.204 0.463 0.566 0.464 ...
 $ concave.points_mean    : num  0.731 0.349 0.636 0.523 0.518 ...
 $ symmetry_mean          : num  0.686 0.38 0.51 0.776 0.378 ...
 $ fractal_dimension_mean : num  0.606 0.141 0.211 1 0.187 ...
 $ radius_se              : num  0.356 0.156 0.23 0.139 0.234 ...
 $ texture_se             : num  0.1205 0.0826 0.0943 0.1759 0.0931 ...
 $ perimeter_se           : num  0.369 0.124 0.18 0.127 0.221 ...
 $ area_se                : num  0.2738 0.1257 0.1629 0.0382 0.1637 ...
 $ smoothness_se          : num  0.159 0.119 0.151 0.251 0.332 ...
 $ compactness_se         : num  0.3514 0.0813 0.284 0.5432 0.1679 ...
 $ concavity_se           : num  0.1357 0.047 0.0968 0.143 0.1436 ...
 $ concave.points_se      : num  0.301 0.254 0.39 0.354 0.357 ...
 $ symmetry_se            : num  0.3116 0.0845 0.2057 0.7281 0.1362 ...
 $ fractal_dimension_se   : num  0.183 0.0911 0.127 0.2872 0.1458 ...
 $ radius_worst           : num  0.621 0.607 0.556 0.248 0.52 ...
 $ texture_worst          : num  0.142 0.304 0.36 0.386 0.124 ...
 $ perimeter_worst        : num  0.668 0.54 0.508 0.241 0.507 ...
 $ area_worst             : num  0.451 0.435 0.375 0.094 0.342 ...
 $ smoothness_worst       : num  0.601 0.348 0.484 0.915 0.437 ...
 $ compactness_worst      : num  0.619 0.155 0.385 0.814 0.172 ...
 $ concavity_worst        : num  0.569 0.193 0.36 0.549 0.319 ...
 $ concave.points_worst   : num  0.912 0.639 0.835 0.885 0.558 ...
 $ symmetry_worst         : num  0.598 0.234 0.404 1 0.158 ...
 $ fractal_dimension_worst: num  0.419 0.223 0.213 0.774 0.143 ...
> wbcd_test = wbcd_n[470:569,]
> wbcd_train = wbcd_n[1:469,]
> wbcd_train_labels = wbcd[1:469, 1]
> wbcd_test_labels = wbcd[470:569, 1]
> install.packages("class")
Installing package into �C:/Users/NISHANT/Documents/R/win-library/3.2�
(as �lib� is unspecified)
--- Please select a CRAN mirror for use in this session ---
trying URL 'https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/contrib/3.2/class_7.3-14.zip'
Content type 'application/zip' length 100092 bytes (97 KB)
downloaded 97 KB

package �class� successfully unpacked and MD5 sums checked

The downloaded binary packages are in
        C:\Users\NISHANT\AppData\Local\Temp\Rtmpq6AiC4\downloaded_packages
> library( class )
Warning message:
package �class� was built under R version 3.2.5 
> wbcd_pred = knn( train  = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k = 3)
> wbcd_pred = knn( train  = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k = 21)
> install.packages("gmodels")
Installing package into �C:/Users/NISHANT/Documents/R/win-library/3.2�
(as �lib� is unspecified)
also installing the dependencies �gtools�, �gdata�

trying URL 'https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/contrib/3.2/gtools_3.5.0.zip'
Content type 'application/zip' length 143976 bytes (140 KB)
downloaded 140 KB

trying URL 'https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/contrib/3.2/gdata_2.17.0.zip'
Content type 'application/zip' length 1177987 bytes (1.1 MB)
downloaded 1.1 MB

trying URL 'https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/contrib/3.2/gmodels_2.16.2.zip'
Content type 'application/zip' length 73923 bytes (72 KB)
downloaded 72 KB

package �gtools� successfully unpacked and MD5 sums checked
package �gdata� successfully unpacked and MD5 sums checked
package �gmodels� successfully unpacked and MD5 sums checked

The downloaded binary packages are in
        C:\Users\NISHANT\AppData\Local\Temp\Rtmpq6AiC4\downloaded_packages

> library(gmodels)
Warning message:
package �gmodels� was built under R version 3.2.5 
> CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq=FALSE)
Error in CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = FALSE) : 
  object 'wbcd_test_pred' not found
> CrossTable(x = wbcd_test_labels, y = wbcd_pred, prop.chisq=FALSE)

 
   Cell Contents
|-------------------------|
|                       N |
|           N / Row Total |
|           N / Col Total |
|         N / Table Total |
|-------------------------|

 
Total Observations in Table:  100 

 
                 | wbcd_pred 
wbcd_test_labels |    Benign | Malignant | Row Total | 
-----------------|-----------|-----------|-----------|
          Benign |        77 |         0 |        77 | 
                 |     1.000 |     0.000 |     0.770 | 
                 |     0.975 |     0.000 |           | 
                 |     0.770 |     0.000 |           | 
-----------------|-----------|-----------|-----------|
       Malignant |         2 |        21 |        23 | 
                 |     0.087 |     0.913 |     0.230 | 
                 |     0.025 |     1.000 |           | 
                 |     0.020 |     0.210 |           | 
-----------------|-----------|-----------|-----------|
    Column Total |        79 |        21 |       100 | 
                 |     0.790 |     0.210 |           | 
-----------------|-----------|-----------|-----------|

 
> 