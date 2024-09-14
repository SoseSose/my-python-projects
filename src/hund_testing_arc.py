from data_process.arc.arc_preprocess import ArcTaskSet
import pyperclip


if __name__ == "__main__":
    arc_task_set = ArcTaskSet()
    arc_task_set = arc_task_set.path_to_arc_task("dataset/Selective-Arc/original_arc/training")
    # for task in arc_task_set:
    task = arc_task_set[2]
    print(task.name)
    print(task.question)
    print(task.test_output)
    pyperclip.copy(task.question)


    
