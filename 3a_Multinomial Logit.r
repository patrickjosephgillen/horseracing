# This script takes less than 1 minute to execute

# Reference:  Statistical Models of Horse Racing Outcomes Using R
#             Dr Alun Owen, Coventry University, UK
#             http://www2.stat-athens.aueb.gr/~jbn/conferences/MathSport_presentations/TRACK%20B/B3%20-%20Sports%20Modelling%20and%20Prediction/AlunOwen_Horse_Racing.pdflibrary("mlogit")

library("dplyr")
library("mlogit")

setwd(dirname(sys.frame(1)$ofile))
train_data <- read.csv(file.path("data\\runners_train.csv"))

train_data$win <- ifelse(train_data$win, TRUE, FALSE)

# create horse ref's ("choices") for each race
train_data <- train_data %>% group_by(race_id) %>%
    reframe(horse.ref = seq_len(n()), runner_id = runner_id, runners = n(), winners = max(win)) %>%
    as.data.frame() %>%
    right_join(train_data, by = c("race_id", "runner_id"))

# ---------
# Fit model
# ---------

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

write.csv(data.frame(feature=labels(coefficients(m)), coefficient=coefficients(m)), "models\\multinomial_logit_coefficients.csv", row.names=F)

# --------------
# Diagnose model
# --------------