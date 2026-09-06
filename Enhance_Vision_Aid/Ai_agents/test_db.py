from database import add_task, get_tasks, update_task, delete_task

# Add tasks
add_task("Complete voice assistant integration")
add_task("Write project documentation")

# Fetch and display tasks
print("Tasks:", get_tasks())

# Update a task
update_task(1, "completed")
print("Updated Tasks:", get_tasks())

# Delete a task
delete_task(2)
print("Remaining Tasks:")
gt=get_tasks()
print(type(gt))