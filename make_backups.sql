CREATE TABLE my_races_bck LIKE my_races;
INSERT INTO my_races_bck SELECT * FROM my_races;

CREATE TABLE my_runners_bck LIKE my_runners;
INSERT INTO my_runners_bck SELECT * FROM my_runners;