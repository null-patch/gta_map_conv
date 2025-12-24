__all__ = [
    "ConversionPipeline",
    "ConversionStats",
    "BatchProcessor",
    "ProjectManager",
    "SceneObject",
]

def __getattr__(name):
    if name in ("ConversionPipeline", "ConversionStats", "BatchProcessor"):
        from .conversion_pipeline import ConversionPipeline, ConversionStats, BatchProcessor
        return {"ConversionPipeline": ConversionPipeline, "ConversionStats": ConversionStats, "BatchProcessor": BatchProcessor}[name]
    if name == "ProjectManager":
        from .project_manager import ProjectManager
        return ProjectManager
    if name == "SceneObject":
        from .models import SceneObject
        return SceneObject
    raise AttributeError(name)