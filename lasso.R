df_all <- read.csv("data_malignant.csv")
sapply(df_all, function(x) sum(is.na(x)))
data <- na.omit(data)
dimension <- dim(df_all)
dimension

col_name <- names(df_all)
col_name[1:100]
#分割训练集和验证集
set.seed(1234)
training_index <- sample(1:150, 0.7 * 150, replace = F)
validation_index  <-  c(1:150)[-training_index]

training_index
validation_index

training_set <- df_all[training_index, ]
validation_set <- df_all[validation_index, ]

train_label <- training_set[, 1]
valid_label <- validation_set[, 1]

train_set_nolabel <- training_set[-1]
dim(training_set)
dim(validation_set)
#对特征进行正态性检验
norm_result <- apply(training_set, 2, function(x) shapiro.test(x)$p.value)
norm_feature <- training_set[which(norm_result >= 0.05)]
cor_nor <- cor(norm_feature, method = "pearson")
#对相关系数矩阵进行修改
cor_all <- cor(training_set, method = "spearman")
num_nor <- dim(cor_nor)[1]
cor_all[1:num_nor, 1:num_nor] <- cor_nor
cor_all[upper.tri(cor_all)] <- 0
diag(cor_all) <- 0
#选择相关性较低的特征
data_reduce = training_set[, !apply(cor_all, 2, function(x) any(abs(x) > 0.9))]
dim(data_reduce)
View(data_reduce)
#使用LASSO算法进行特征选择
library(glmnet)

cv_x <- as.matrix(data_reduce)
cv_y <- train_label
set.seed(1)
lasso_selection <- cv.glmnet(x = cv_x,
                             y = cv_y, 
                             family = "binomial", 
                             type.measure = "deviance",
                             alpha = 1, 
                             nfolds = 5)
#绘制lasso 路径图
par(font.lab = 2, mfrow = c(2,1), mar = c(4.5, 5, 3, 2))
plot(x = lasso_selection, las = 1, xlab = "Log(lambda)") 

nocv_lasso <- glmnet(x = cv_x, y = cv_y, family = "binomial", alpha = 1)
plot(nocv_lasso, xvar = "lambda", las = 1, lwd = 2, xlab = "Log(lambda)") 
abline(v = log(lasso_selection$lambda.min), lwd = 1, lty = 3, col = "black")
#获取Lasso选择的特征及其系数
coefPara <- coef(object = lasso_selection, s = "lambda.min")
lasso_values <- as.data.frame( which(coefPara != 0, arr.ind = T))
lasso_names <- rownames(lasso_values)[-1]
Lasso_coef <- data.frame(Feature = rownames(lasso_values),
                         Coef = coefPara[which(coefPara != 0, arr.ind = T)])
Lasso_coef
#根据Lasso模型选择的特征子集，从原始数据集提取对应的训练集和验证集
train_set_lasso <- data.frame(cv_x)[lasso_names]
valid_set_lasso <- validation_set[names(train_set_lasso)]
Data_all = as.matrix(rbind(train_set_lasso, valid_set_lasso))
#提取系数矩阵和截距
xn = nrow(Data_all)	
yn = ncol(Data_all)
beta <- as.matrix( coefPara[ which(coefPara != 0),])
betai.Matrix <- as.matrix( beta[-1])
beta0_Matrix <-	matrix(beta[1], xn, 1 )
#计算Lasso 模型得分
Radcore_Matrix <- Data_all %*% betai.Matrix + beta0_Matrix
radscore_all<- as.numeric(Radcore_Matrix)
Radscore_train <- radscore_all[1:nrow(train_set_lasso)]
Radscore_valid <- radscore_all[(nrow(train_set_lasso)+1):xn]
Radscore_train
Radscore_valid

model_log <- glm(formula = train_label~.,data= train_set_lasso,family = "binomial",maxit = 1000)
summary(model_log)

pred_train_log <- predict(object= model_log,newdata = train_set_lasso, type="response")
pred_valid_log <- predict(object= model_log,newdata = valid_set_lasso, type="response")

library(pROC)
roc_train_log <- roc(train_label, pred_train_log ,levels= c(0,1),direction= "<")
roc_valid_log <- roc(valid_label, pred_valid_log ,levels= c(0,1),direction= "<")

roc_valid_log$auc