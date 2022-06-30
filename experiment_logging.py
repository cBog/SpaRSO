from datetime import datetime
import git
import os
import pickle
import pprint

import numpy as np
from matplotlib import pyplot as plt

class Logger:
  LOGS_PATH = "logs"
  LOG_INDEX_FILE_PATH = os.path.join(LOGS_PATH, "log_index.txt")

  def __init__(self, description, args):
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty():
      print("there are uncommitted changes, cannot execute")
      import pdb; pdb.set_trace()
    assert not repo.is_dirty(), "there are uncommitted changes, cannot execute"
    self.sha = repo.head.object.hexsha
    self.description = description

    # get id and date time
    now = datetime.now()
    self.time_id = now.strftime("%Y%m%d_%H%M_{self.sha}")
    self.time_str = now.strftime("%d/%m/%Y %H:%M:%S")

    # make directory in logs/id
    self.id_dir_path = os.path.join(self.LOGS_PATH, self.time_id)
    os.mkdir(self.id_dir_path)

    # save id: (hash) description to logs/log_index.txt
    log_index_file = open(self.LOG_INDEX_FILE_PATH, 'a')
    log_index_file.write(f"{self.time_id} : ({self.sha}) {description}\n")
    log_index_file.close()

    # save logs/id/info.txt information with args, description, commit hash, date
    self.info_log_path = os.path.join(self.id_dir_path,"info_summary.txt")
    info_file = open(self.info_log_path, mode='a')
    lines = []

    lines.append(f"### LOG of {self.sha} on {self.time_str}")
    lines.append(f"id : {self.time_id}")
    lines.append(f"{description}")
    lines.append("")
    lines.append(f"ARGS: ")
    lines.append(pprint.pformat(args, sort_dicts=True))
    lines.append("")

    info_file.writelines(lines)
    info_file.close()

    # create logs/id/output.txt
    self.output_log_path = os.path.join(self.id_dir_path,"output.txt")
    self.output_log_file = open(self.output_log_path, mode='a')
  
  # TODO: get this working
  # def __del__(self):
  #   import pdb; pdb.set_trace()
  #   if self.output_log_file is None and not self.output_log_file.closed:
  #     self.output_log_file.close()

  def log_model_summary(self, model):
    # add model info to logs/id/info.txt
    def myprint(s):
      with open(self.info_log_path,'a') as f:
        print(s, file=f)

    model.summary(print_fn=myprint)

  def log_eval_results(self, test_loss, test_acc):
    report_string = f"VAL LOSS = {test_loss}, VAL ACC = {test_acc}"
    self.log(report_string)
    with open(self.info_log_path,'a') as f:
      print("FINAL RESULTS:", file=f)
      print(report_string, file=f)

  def log(self, text, log_type_str="default"):
    # append a line to the log file
    now = datetime.now()
    date_time_now_str = now.strftime("%H:%M:%S")
    print(f"{date_time_now_str} [{log_type_str}]: ", file=self.output_log_file)
  
  def save(self, item, name):
    if isinstance(item, (np.ndarray, np.generic)):
      file_name = f"{os.path.join(self.id_dir_path,name)}.np"
      np.save(file_name, item)
    elif isinstance(item, pyplot.Figure):
      file_name = f"{os.path.join(self.id_dir_path,name)}.png"
      item.savefig(file_name)
    else:
      file_name = f"{os.path.join(self.id_dir_path,name)}.pkl"
      file = open(file_name, mode='w')
      pickle.dump(item, file)
      file.close()

    self.log(f"saved file {file_name}", "meta")