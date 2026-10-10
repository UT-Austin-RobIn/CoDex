"""The app's stage status bar: progress and summaries read off stage logs."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import app  # noqa: E402


class StatusTest(unittest.TestCase):
    def test_policy_evaluation_does_not_retarget_the_saved_task(self):
        with mock.patch.object(app, "need_scene", return_value="/example/scene"), \
                mock.patch.object(app, "goal_file", return_value="/example/changed_goal.json"), \
                mock.patch.object(app, "stream", return_value=iter(())) as stream:
            list(app.run_evaluation("scene", "/example/run", 128))
        command = stream.call_args.args[1]
        self.assertNotIn("--goal-method", command)
        self.assertNotIn("--goal", command)
        self.assertEqual(command[command.index("--run-dir") + 1], "/example/run")

    def test_goal_rounds(self):
        self.assertEqual(app.goal_progress_text("Scene loaded"), "Preparing the scene")
        self.assertEqual(app.goal_progress_text("Iter 1/5\n"), "Round 1 of 5: rendering candidates")
        self.assertEqual(app.goal_progress_text("Iter 1/5\nEval prompt:\n"),
                         "Round 1 of 5: the VLM is scoring the candidates")
        self.assertEqual(app.goal_progress_text("Iter 1/5\nID1: score=0.2\nIter 2/5\n"),
                         "Round 2 of 5: the VLM is proposing candidates")

    def test_grasps(self):
        self.assertEqual(app.grasp_progress("Optimising 2048 candidates (300 steps) ...\n  step  200 E_med=1"),
                         "Optimising candidate grasps: step 200 of 300")
        self.assertIn("round 2 of 3", app.grasp_progress(
            "  halving stage 1: 192 candidates x 8 rollouts; best success=0.30 lift=0.70 (n=10)"))
        log = ("  grasp_1: actuation finger 0, simulated success 33%, lift 57% over 42 rollouts\n"
               "  grasp_2: actuation finger 0, simulated success 14%, lift 79% over 42 rollouts")
        self.assertEqual(app.grasp_summary(log), "2 grasps; the best succeeds in 33% of simulated trials.")

    def test_evaluation_and_training(self):
        self.assertEqual(app.eval_progress("Evaluating grasp_1, zero action: 32 episodes, seed 0"),
                         "Simulating 32 episodes")
        self.assertEqual(app.eval_summary("  success: 94%  goal: 99%  stable: 99%"),
                         "Task success 94%, at the goal 99%.")
        log = "|    total_timesteps | 12288 |\nSuccess rate: 44%"
        self.assertEqual(app.training_progress(log, 400000), "12,288 of 400,000 steps (3%); last evaluation 44%")
        self.assertEqual(app.training_summary(log), "Trained 12,288 steps; last evaluation 44%.")

    def test_status_bar_states(self):
        self.assertIn("Running <time>1:05</time>", app.run_status("running", "Round 1 of 5", 65))
        self.assertIn("Done in 4:10", app.run_status("done", "", 250))
        self.assertIn("Failed after 0:25", app.run_status("failed", "the OpenAI account has no credits left", 25))
        self.assertEqual(app.failure_reason("x\nopenai.RateLimitError: Error code: 429 insufficient_quota"),
                         "the OpenAI account has no credits left")



class GoalRoundsTest(unittest.TestCase):
    def test_rounds_are_read_off_the_log_newest_first(self):
        log = ("Iter 1/2\nEval prompt:\nID1: params=(0.1, 0, 0, 0), score=0.200, pen=0.000, desc=too far\n"
               "ID2: params=(0.0, 0, 0, 0), score=0.800, pen=0.000, desc=aims at the bowl\n"
               "Iter 2/2\n")
        out = app.goal_rounds("out/scenes/does_not_exist", "vlm_cem_dir", log)
        self.assertLess(out.index("Round 2"), out.index("Round 1: best #2, score 0.80"))
        self.assertIn("aims at the bowl", out)
        self.assertEqual(app.goal_rounds("out/scenes/does_not_exist", "vlm_cem_dir"), "")


if __name__ == "__main__":
    unittest.main()
