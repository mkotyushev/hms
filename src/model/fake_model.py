from lightning.pytorch.demos.boring_classes import DemoModel


class FakeModel(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using FakeModel", "⚡")
        return super().configure_optimizers()
