/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class edu_uw_Taggerflow */

#ifndef _Included_edu_uw_Taggerflow
#define _Included_edu_uw_Taggerflow
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     edu_uw_Taggerflow
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_edu_uw_Taggerflow_close
  (JNIEnv *, jclass, jlong);

/*
 * Class:     edu_uw_Taggerflow
 * Method:    initialize
 * Signature: (Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_edu_uw_Taggerflow_initialize
  (JNIEnv *, jclass, jstring, jstring);

/*
 * Class:     edu_uw_Taggerflow
 * Method:    predictPacked
 * Signature: ([BJ)[B
 */
JNIEXPORT jbyteArray JNICALL Java_edu_uw_Taggerflow_predictPacked___3BJ
  (JNIEnv *, jclass, jbyteArray, jlong);

/*
 * Class:     edu_uw_Taggerflow
 * Method:    predictPacked
 * Signature: (Ljava/lang/String;IJ)[B
 */
JNIEXPORT jbyteArray JNICALL Java_edu_uw_Taggerflow_predictPacked__Ljava_lang_String_2IJ
  (JNIEnv *, jclass, jstring, jint, jlong);

#ifdef __cplusplus
}
#endif
#endif
