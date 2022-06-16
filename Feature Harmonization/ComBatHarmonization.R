require(data.table)


T1 <- read.csv("Radiomics_cohort1.csv",header = T)
T2 <- read.csv(Radiomics_cohort2.csv",header = T)
T1<-T1[,2:dim(T1)[2]]
T2<-T2[,2:dim(T2)[2]]
dim(T1);dim(T2)
T12 <- rbind(T1,T2);dim(T12)
T12t<-as.data.frame(t(T12))
names(T12t)<-c(1:(dim(T1)[1]+dim(T2)[1]))
str(T12t)
dim(T12t)


library(sva)


Thead <- read.csv("../data/HEAD.csv",header = T,row.names = 1)
batch = Thead$batch
modcombat = model.matrix(~1, data=Thead)

combat_edata = ComBat(dat=as.matrix(T12t), batch=batch, mod=modcombat, par.prior=FALSE)
combat_edatat<-as.data.frame(t(combat_edata))
str(combat_edatat)
head(combat_edatat)


cohort1<-combat_edatat[1:dim(T1)[1],]
cohort2<-combat_edatat[(dim(T1)[1]+1):(dim(T1)[1]+dim(T2)[1]),]
write.csv(cohort1, quote=FALSE, file="../data/_harmonized_cohort1.csv", row.names=FALSE)
write.csv(cohort2, quote=FALSE, file="../data/_harmonized_cohort2.csv", row.names=FALSE)


