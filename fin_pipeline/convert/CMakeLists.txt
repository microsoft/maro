PROJECT (converter)

SET (CONVERTER_SRC converter.c)


ADD_EXECUTABLE(converter ${CONVERTER_SRC})

TARGET_LINK_LIBRARIES(converter pthread)