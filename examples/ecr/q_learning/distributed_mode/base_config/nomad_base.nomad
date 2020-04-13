job "maro" {
  region = "global"
  datacenters = ["dc1"]
  type = "service"

  group "learner" {
    count = 1

    task "learner" {
      driver = "docker"

      config {
        command = "python environment_runner.py"

        args = ["LOG_LEVEL", "PROGRESS", "GROUP", "test_ep500_1wDbis", "COMPTYPE", "learner", "COMPID", "4"]
      }

      resources {
        cpu    = 500
        memory = 256
      }
    }
  }
}