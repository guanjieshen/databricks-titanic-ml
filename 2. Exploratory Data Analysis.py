# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT * FROM emagine.titanic_train

# COMMAND ----------

# MAGIC %md We can also install `bamboolib` to get access to UI based data transformation options.

# COMMAND ----------

# MAGIC %pip install bamboolib

# COMMAND ----------

import bamboolib as bam
bam
