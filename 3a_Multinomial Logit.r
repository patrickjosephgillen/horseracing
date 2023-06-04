# This script takes less than 1 minute to execute

# Reference:  Statistical Models of Horse Racing Outcomes Using R
#             Dr Alun Owen, Coventry University, UK
#             http://www2.stat-athens.aueb.gr/~jbn/conferences/MathSport_presentations/TRACK%20B/B3%20-%20Sports%20Modelling%20and%20Prediction/AlunOwen_Horse_Racing.pdflibrary("mlogit")

library("dplyr")
library("mlogit")

setwd(dirname(sys.frame(1)$ofile))
train_data <- read.csv(file.path("data\\runners_train.csv"))

# Read only the first row to get the column names
column_names <- names(read.csv(file.path("data\\races_train.csv"), nrows = 1))

# Define the column names you're interested in
columns_to_read <- c("race_id", "course_Kempton", "course_Lingfield", "course_Southwell", "course_Wolverhampton", "class_5", "class_4", "class_3", "class_2", "class_1", "gt1mi")

# Get the indices of the columns you're interested in
column_indices <- which(column_names %in% columns_to_read)

# Read only the columns you're interested in
race_train_data <- read.csv(file.path("data\\races_train.csv"), colClasses = ifelse(1:length(column_names) %in% column_indices, NA, "NULL"))

train_data <- merge(train_data, race_train_data, by = "race_id")

train_data$win <- ifelse(train_data$win, TRUE, FALSE)

# create horse ref's ("choices") for each race
train_data <- train_data %>% group_by(race_id) %>%
    reframe(horse.ref = seq_len(n()), runner_id = runner_id, runners = n(), winners = max(win)) %>%
    as.data.frame() %>%
    right_join(train_data, by = c("race_id", "runner_id"))

# -------------------
# Fit Alun Owen model
# -------------------

select_cols <- c("race_id", "horse.ref", "runner_id", "win",
    "age", "sire_sr", "dam_sr", "trainer_sr", "daysLTO",
    "position1_1", "position1_2", "position1_3", "position1_4",
    "position2_1", "position2_2", "position2_3", "position2_4",
    "position3_1", "position3_2", "position3_3", "position3_4",
    "entire", "gelding",
    "blinkers", "visor", "cheekpieces", "tonguetie")

h_dat <- mlogit.data(data = train_data[select_cols],
    choice = "win", chid.var = "race_id", alt.var = "horse.ref",
    shape = "long")

m <- mlogit(win ~
    age + sire_sr + dam_sr + trainer_sr + daysLTO +
    position1_1 + position1_2 + position1_3 + position1_4 +
    position2_1 + position2_2 + position2_3 + position2_4 +
    position3_1 + position3_2 + position3_3 + position3_4 +
    entire + gelding +
    blinkers + visor + cheekpieces + tonguetie
    | 0 | 0, data = h_dat)

print(summary(m))

m <- mlogit(win ~
    age + trainer_sr + daysLTO +
    position1_1 + position1_2 + position1_3 + position1_4 +
    position2_1 + position2_2 + position2_3 + position2_4 +
    position3_1 + position3_2 + position3_3 + position3_4 +
    entire + gelding +
    blinkers + cheekpieces + tonguetie
    | 0 | 0, data = h_dat)

print(summary(m))

write.csv(data.frame(feature=labels(coefficients(m)), coefficient=coefficients(m)), "models\\AlunOwen_multinomial_logit_coefficients.csv", row.names=F)

# -------------
# Fit MG1 model
# -------------

# Calculate interactions

train_data$sr_lifetime_class_5 <- train_data$sr_lifetime_class_5 * train_data$class_5
train_data$sr_lifetime_class_4 <- train_data$sr_lifetime_class_4 * train_data$class_4
train_data$sr_lifetime_class_3 <- train_data$sr_lifetime_class_3 * train_data$class_3
train_data$sr_lifetime_class_2 <- train_data$sr_lifetime_class_2 * train_data$class_2
train_data$sr_lifetime_class_1 <- train_data$sr_lifetime_class_1 * train_data$class_1

train_data$jockey_sr_KEM <- train_data$jockey_sr_KEM * train_data$course_Kempton
train_data$jockey_sr_LIN <- train_data$jockey_sr_LIN * train_data$course_Lingfield
train_data$jockey_sr_SOU <- train_data$jockey_sr_SOU * train_data$course_Southwell
train_data$jockey_sr_WOL <- train_data$jockey_sr_WOL * train_data$course_Wolverhampton

train_data$trainer_sr_KEM <- train_data$trainer_sr_KEM * train_data$course_Kempton
train_data$trainer_sr_LIN <- train_data$trainer_sr_LIN * train_data$course_Lingfield
train_data$trainer_sr_SOU <- train_data$trainer_sr_SOU * train_data$course_Southwell
train_data$trainer_sr_WOL <- train_data$trainer_sr_WOL * train_data$course_Wolverhampton

train_data$pos_prior1_KEM_le1mi <- train_data$pos_prior1_KEM_le1mi * train_data$course_Kempton
train_data$pos_prior2_KEM_le1mi <- train_data$pos_prior2_KEM_le1mi * train_data$course_Kempton
train_data$pos_prior3_KEM_le1mi <- train_data$pos_prior3_KEM_le1mi * train_data$course_Kempton

train_data$pos_prior1_LIN_le1mi <- train_data$pos_prior1_LIN_le1mi * train_data$course_Lingfield
train_data$pos_prior2_LIN_le1mi <- train_data$pos_prior2_LIN_le1mi * train_data$course_Lingfield
train_data$pos_prior3_LIN_le1mi <- train_data$pos_prior3_LIN_le1mi * train_data$course_Lingfield

train_data$pos_prior1_SOU_le1mi <- train_data$pos_prior1_SOU_le1mi * train_data$course_Southwell
train_data$pos_prior2_SOU_le1mi <- train_data$pos_prior2_SOU_le1mi * train_data$course_Southwell
train_data$pos_prior3_SOU_le1mi <- train_data$pos_prior3_SOU_le1mi * train_data$course_Southwell

train_data$pos_prior1_WOL_le1mi <- train_data$pos_prior1_WOL_le1mi * train_data$course_Wolverhampton
train_data$pos_prior2_WOL_le1mi <- train_data$pos_prior2_WOL_le1mi * train_data$course_Wolverhampton
train_data$pos_prior3_WOL_le1mi <- train_data$pos_prior3_WOL_le1mi * train_data$course_Wolverhampton

train_data$pos_prior1_KEM_gt1mi <- train_data$pos_prior1_KEM_gt1mi * train_data$course_Kempton * train_data$gt1mi
train_data$pos_prior2_KEM_gt1mi <- train_data$pos_prior2_KEM_gt1mi * train_data$course_Kempton * train_data$gt1mi
train_data$pos_prior3_KEM_gt1mi <- train_data$pos_prior3_KEM_gt1mi * train_data$course_Kempton * train_data$gt1mi

train_data$pos_prior1_LIN_gt1mi <- train_data$pos_prior1_LIN_gt1mi * train_data$course_Lingfield * train_data$gt1mi
train_data$pos_prior2_LIN_gt1mi <- train_data$pos_prior2_LIN_gt1mi * train_data$course_Lingfield * train_data$gt1mi
train_data$pos_prior3_LIN_gt1mi <- train_data$pos_prior3_LIN_gt1mi * train_data$course_Lingfield * train_data$gt1mi

train_data$pos_prior1_SOU_gt1mi <- train_data$pos_prior1_SOU_gt1mi * train_data$course_Southwell * train_data$gt1mi
train_data$pos_prior2_SOU_gt1mi <- train_data$pos_prior2_SOU_gt1mi * train_data$course_Southwell * train_data$gt1mi
train_data$pos_prior3_SOU_gt1mi <- train_data$pos_prior3_SOU_gt1mi * train_data$course_Southwell * train_data$gt1mi

train_data$pos_prior1_WOL_gt1mi <- train_data$pos_prior1_WOL_gt1mi * train_data$course_Wolverhampton * train_data$gt1mi
train_data$pos_prior2_WOL_gt1mi <- train_data$pos_prior2_WOL_gt1mi * train_data$course_Wolverhampton * train_data$gt1mi
train_data$pos_prior3_WOL_gt1mi <- train_data$pos_prior3_WOL_gt1mi * train_data$course_Wolverhampton * train_data$gt1mi

select_cols <- c("race_id", "horse.ref", "runner_id", "win",
    "sr_30d", "sr_lifetime_class_5", "sr_lifetime_class_4", "sr_lifetime_class_3", "sr_lifetime_class_2", "sr_lifetime_class_1",
    "jockey_sr_KEM", "jockey_sr_LIN", "jockey_sr_SOU", "jockey_sr_WOL",
    "trainer_sr_KEM", "trainer_sr_LIN", "trainer_sr_SOU", "trainer_sr_WOL",
    "jockey_sr_30d", "trainer_sr_30d",
    "pos_prior1_KEM_le1mi", "pos_prior2_KEM_le1mi", "pos_prior3_KEM_le1mi",
    "pos_prior1_LIN_le1mi", "pos_prior2_LIN_le1mi", "pos_prior3_LIN_le1mi",
    "pos_prior1_SOU_le1mi", "pos_prior2_SOU_le1mi", "pos_prior3_SOU_le1mi", 
    "pos_prior1_WOL_le1mi", "pos_prior2_WOL_le1mi", "pos_prior3_WOL_le1mi",
    "pos_prior1_KEM_gt1mi", "pos_prior2_KEM_gt1mi", "pos_prior3_KEM_gt1mi",
    "pos_prior1_LIN_gt1mi", "pos_prior2_LIN_gt1mi", "pos_prior3_LIN_gt1mi",
    "pos_prior1_SOU_gt1mi", "pos_prior2_SOU_gt1mi", "pos_prior3_SOU_gt1mi",
    "pos_prior1_WOL_gt1mi", "pos_prior2_WOL_gt1mi", "pos_prior3_WOL_gt1mi",
    "position1_1", "position1_2", "position1_3", "position1_4",
    "position2_1", "position2_2", "position2_3", "position2_4",
    "position3_1", "position3_2", "position3_3", "position3_4")

h_dat <- mlogit.data(data = train_data[select_cols],
    choice = "win", chid.var = "race_id", alt.var = "horse.ref",
    shape = "long")

m <- mlogit(win ~
    sr_30d + sr_lifetime_class_5 + sr_lifetime_class_4 + sr_lifetime_class_3 + sr_lifetime_class_2 + sr_lifetime_class_1 +
    jockey_sr_KEM + jockey_sr_LIN + jockey_sr_SOU + jockey_sr_WOL +
    trainer_sr_KEM + trainer_sr_LIN + trainer_sr_SOU + trainer_sr_WOL +
    jockey_sr_30d + trainer_sr_30d +
    pos_prior1_KEM_le1mi + pos_prior2_KEM_le1mi + pos_prior3_KEM_le1mi +
    pos_prior1_LIN_le1mi + pos_prior2_LIN_le1mi + pos_prior3_LIN_le1mi +
    pos_prior1_SOU_le1mi + pos_prior2_SOU_le1mi + pos_prior3_SOU_le1mi + 
    pos_prior1_WOL_le1mi + pos_prior2_WOL_le1mi + pos_prior3_WOL_le1mi +
    pos_prior1_KEM_gt1mi + pos_prior2_KEM_gt1mi + pos_prior3_KEM_gt1mi +
    pos_prior1_LIN_gt1mi + pos_prior2_LIN_gt1mi + pos_prior3_LIN_gt1mi +
    pos_prior1_SOU_gt1mi + pos_prior2_SOU_gt1mi + pos_prior3_SOU_gt1mi + 
    pos_prior1_WOL_gt1mi + pos_prior2_WOL_gt1mi + pos_prior3_WOL_gt1mi +
    position1_1 + position1_2 + position1_3 + position1_4 +
    position2_1 + position2_2 + position2_3 + position2_4 +
    position3_1 + position3_2 + position3_3 + position3_4
    | 0 | 0, data = h_dat)

print(summary(m))

m <- mlogit(win ~
    sr_30d + sr_lifetime_class_5 + sr_lifetime_class_4 + sr_lifetime_class_3 + sr_lifetime_class_2 + sr_lifetime_class_1 +
    jockey_sr_30d + trainer_sr_30d +
    position1_1 + position1_2 + position1_3 + position1_4
    | 0 | 0, data = h_dat)

print(summary(m))

write.csv(data.frame(feature=labels(coefficients(m)), coefficient=coefficients(m)), "models\\MG1_multinomial_logit_coefficients.csv", row.names=F)

# -----------------
# Fit Amalgum model
# -----------------

select_cols <- c("race_id", "horse.ref", "runner_id", "win",
    "age", "sire_sr", "dam_sr", "trainer_sr", "daysLTO",
    "position1_1", "position1_2", "position1_3", "position1_4",
    "position2_1", "position2_2", "position2_3", "position2_4",
    "position3_1", "position3_2", "position3_3", "position3_4",
    "entire", "gelding",
    "blinkers", "visor", "cheekpieces", "tonguetie",
    "sr_30d", "sr_lifetime_class_5", "sr_lifetime_class_4", "sr_lifetime_class_3", "sr_lifetime_class_2", "sr_lifetime_class_1",
    "jockey_sr_KEM", "jockey_sr_LIN", "jockey_sr_SOU", "jockey_sr_WOL",
    "trainer_sr_KEM", "trainer_sr_LIN", "trainer_sr_SOU", "trainer_sr_WOL",
    "jockey_sr_30d", "trainer_sr_30d",
    "pos_prior1_KEM_le1mi", "pos_prior2_KEM_le1mi", "pos_prior3_KEM_le1mi",
    "pos_prior1_LIN_le1mi", "pos_prior2_LIN_le1mi", "pos_prior3_LIN_le1mi",
    "pos_prior1_SOU_le1mi", "pos_prior2_SOU_le1mi", "pos_prior3_SOU_le1mi", 
    "pos_prior1_WOL_le1mi", "pos_prior2_WOL_le1mi", "pos_prior3_WOL_le1mi",
    "pos_prior1_KEM_gt1mi", "pos_prior2_KEM_gt1mi", "pos_prior3_KEM_gt1mi",
    "pos_prior1_LIN_gt1mi", "pos_prior2_LIN_gt1mi", "pos_prior3_LIN_gt1mi",
    "pos_prior1_SOU_gt1mi", "pos_prior2_SOU_gt1mi", "pos_prior3_SOU_gt1mi",
    "pos_prior1_WOL_gt1mi", "pos_prior2_WOL_gt1mi", "pos_prior3_WOL_gt1mi",
    "position1_1", "position1_2", "position1_3", "position1_4",
    "position2_1", "position2_2", "position2_3", "position2_4",
    "position3_1", "position3_2", "position3_3", "position3_4")

h_dat <- mlogit.data(data = train_data[select_cols],
    choice = "win", chid.var = "race_id", alt.var = "horse.ref",
    shape = "long")

m <- mlogit(win ~
    age + trainer_sr +
    position1_1 + position1_2 + position1_3 + position1_4 +
    position2_1 + position2_2 + position2_3 + position2_4 +
    position3_1 + position3_2 + position3_3 + position3_4 +
    entire + gelding +
    blinkers + cheekpieces + tonguetie +
    sr_30d + sr_lifetime_class_5 + sr_lifetime_class_4 + sr_lifetime_class_3 + sr_lifetime_class_2 + sr_lifetime_class_1 +
    jockey_sr_30d + trainer_sr_30d
    | 0 | 0, data = h_dat)

print(summary(m))

write.csv(data.frame(feature=labels(coefficients(m)), coefficient=coefficients(m)), "models\\Amalgum_multinomial_logit_coefficients.csv", row.names=F)