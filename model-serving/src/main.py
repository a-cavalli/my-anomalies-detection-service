import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

import os
import io
import matplotlib

matplotlib.use('agg')

settings = get_settings()


class MyService(Service):
    """
    My anomalies detection service model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="My anomalies detection service",
            slug="my-anomalies-detection-service",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="dataset",
                    type=[
                        FieldDescriptionType.TEXT_CSV,
                        FieldDescriptionType.TEXT_PLAIN,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.IMAGE_PNG]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.ANOMALY_DETECTION,
                    acronym=ExecutionUnitTagAcronym.ANOMALY_DETECTION
                ),
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.TIME_SERIES,
                    acronym=ExecutionUnitTagAcronym.TIME_SERIES
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/ae-ano-detection/", 
        )
        self._logger = get_logger(settings)

        self._model = tf.keras.models.load_model(
            os.path.join(os.path.dirname(__file__), "..", "anomalies_detection_model.h5")
        )

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        raw = data["dataset"].data
        input_type = data["dataset"].type

        print("Input type: ", str(input_type))

        X_test = pd.read_csv(io.BytesIO(raw))

        # Use the model to reconstruct the original time series data
        reconstructed_X = self._model.predict(X_test)

        # Calculate the reconstruction error for each point in the time series
        reconstruction_error = np.square(X_test - reconstructed_X).mean(axis=1)

        err = X_test
        fig, ax = plt.subplots(figsize=(20, 6))

        a = err.loc[reconstruction_error >= np.mean(reconstruction_error)]  # anomaly
        ax.plot(err, color='blue', label='Normal')
        ax.scatter(a.index, a, color='red', label='Anomaly')
        plt.legend()

        # Save the plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        # Reset the buffer
        buf.seek(0)

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result": TaskData(data=buf.read(), type=FieldDescriptionType.IMAGE_PNG)
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(
                    my_service, engine_url
                )
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """This service detects anomalies in a time series using an autoencoder.

The service expects a CSV file with a single column containing the time series data.

The service returns a plot of the time series with the detected anomalies highlighted in red.
"""
api_summary = """My anomalies detection service detects anomalies in a time series using an autoencoder.
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="My anomalies detection service API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
