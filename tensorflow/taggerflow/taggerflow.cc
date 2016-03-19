#include "tensorflow/taggerflow/taggerflow.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <unordered_map>
#include "tensorflow/taggerflow/tagging.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;
using namespace taggerflow;
namespace pb = google::protobuf;

const static int kMaxTokens = 70;
const static int kNumFeatures = 9;
const static int kBucketSize = 14;
const static int kNumBuckets = kMaxTokens/kBucketSize;

struct MetaSession {
  std::unique_ptr<Session> session;
  std::unordered_map<std::string, int> feature_maps[kNumFeatures];
  int unknown_indexes[kNumFeatures];
  int oor_indexes[kNumFeatures];
  int start_indexes[kNumFeatures];
  int end_indexes[kNumFeatures];
};

template <typename Message>
void jbytes_to_message(jbyteArray buffer, Message *message, JNIEnv *env) {
  jbyte *buffer_elements = env->GetByteArrayElements(buffer, 0);
  int buffer_length = env->GetArrayLength(buffer);
  CHECK(message->ParseFromArray(reinterpret_cast<void*>(buffer_elements), buffer_length));
  env->ReleaseByteArrayElements(buffer, buffer_elements, JNI_ABORT);
}

jbyteArray message_to_jbytes(const pb::MessageLite &message, JNIEnv *env) {
  std::string buffer;
  CHECK(message.SerializeToString(&buffer));
  jbyteArray bytes = env->NewByteArray(buffer.size());
  env->SetByteArrayRegion(bytes, 0, buffer.size(), (jbyte*) buffer.data());
  return bytes;
}

int get_word_feature_index(const std::string& word, int feature_number, const MetaSession& meta_session) {
  std::string lower;
  lower.resize(word.size());
  std::transform(word.begin(),
                 word.end(),
                 lower.begin(),
                 ::tolower);

  const std::unordered_map<std::string, int>& feature_map = meta_session.feature_maps[feature_number];
  auto it = feature_map.find(lower);
  if (it != feature_map.end()) {
    return it->second;
  } else {
    return meta_session.unknown_indexes[feature_number];
  }
}

int get_morpho_feature_index(const std::string& word, unsigned int length, bool is_prefix, int feature_number, const MetaSession& meta_session) {
  if (word.length() >= length) {
    const std::unordered_map<std::string, int>& feature_map = meta_session.feature_maps[feature_number];
    auto it = feature_map.find(is_prefix ? word.substr(0, length) : word.substr(word.length() - length, word.length()));
    if (it != feature_map.end()) {
      return it->second;
    } else {
      return meta_session.unknown_indexes[feature_number];
    }
  } else {
    return meta_session.oor_indexes[feature_number];
  }
}

void read_feature_map(const std::string& spaces_dir, const std::string& filename, std::unordered_map<std::string, int> *feature_map) {
  std::ifstream file(spaces_dir + "/" + filename);
  std::string line;
  int i = 0;
  while (std::getline(file, line)) {
    feature_map->insert(std::pair<std::string, int>(line, i++));
  }
}

void extract_features(const TaggingInput& input, const MetaSession& meta_session, Tensor *indexes, Tensor *num_tokens) {
  TTypes<int32,3>::Tensor indexes_tensor = indexes->tensor<int32,3>();
  TTypes<int64,1>::Tensor num_tokens_tensor = num_tokens->tensor<int64,1>();

  for (int i = 0; i < input.sentence_size(); ++i) {
    int j = 0;
    for (int k = 0; k < kNumFeatures; ++k) {
      indexes_tensor(i,j,k) = meta_session.start_indexes[k];
    }
    ++j;

    const TaggingInputSentence& sentence = input.sentence(i);
    if (sentence.word_size() <= kMaxTokens) {
      for (const std::string &word : sentence.word()) {
        indexes_tensor(i,j,0) = get_word_feature_index(word, 0, meta_session);
        indexes_tensor(i,j,1) = get_morpho_feature_index(word, 1, true, 1, meta_session);
        indexes_tensor(i,j,2) = get_morpho_feature_index(word, 2, true, 2, meta_session);
        indexes_tensor(i,j,3) = get_morpho_feature_index(word, 3, true, 3, meta_session);
        indexes_tensor(i,j,4) = get_morpho_feature_index(word, 4, true, 4, meta_session);
        indexes_tensor(i,j,5) = get_morpho_feature_index(word, 1, false, 5, meta_session);
        indexes_tensor(i,j,6) = get_morpho_feature_index(word, 2, false, 6, meta_session);
        indexes_tensor(i,j,7) = get_morpho_feature_index(word, 3, false, 7, meta_session);
        indexes_tensor(i,j,8) = get_morpho_feature_index(word, 4, false, 8, meta_session);
        ++j;
      }
    }
    for (int k = 0; k < kNumFeatures; ++k) {
      indexes_tensor(i,j,k) = meta_session.end_indexes[k];
    }
    ++j;
    num_tokens_tensor(i) = j;
    for (; j < kMaxTokens + 2; ++j) {
      for (int k = 0; k < kNumFeatures; ++k) {
        indexes_tensor(i,j,k) = 0;
      }
    }
  }
}

void read_sentences(const char * filename, TaggingInput *input) {
  std::ifstream file(filename);
  std::string line, buf;
  while (std::getline(file, line)) {
    TaggingInputSentence *sentence = input->add_sentence();
    std::stringstream ss(line);
    while (ss >> buf) {
      sentence->add_word(buf);
    }
  }
}

void parse_input(const TaggingInput& input, MetaSession *meta_session, TaggingResult *result) {
  Tensor indexes(DT_INT32, TensorShape({ input.sentence_size(), kMaxTokens + 2, kNumFeatures }));
  Tensor num_tokens(DT_INT64, TensorShape({ input.sentence_size() }));

  std::vector<std::pair<std::string, Tensor>> inputs = {
    { "frozen/model/inputs/x", indexes },
    { "frozen/model/inputs/num_tokens", num_tokens }
  };

  extract_features(input, *meta_session, &indexes, &num_tokens);

  std::vector<Tensor> outputs;

  // Run the session, evaluating the prediction scores from the graph.
  TF_CHECK_OK(meta_session->session->Run(inputs, { "frozen/model/prediction/scores" }, {}, &outputs));

  TTypes<float,3>::Tensor scores = outputs[0].tensor<float,3>();
  TTypes<int64>::Vec num_tokens_vec = num_tokens.vec<int64>();
  int num_supertags = outputs[0].shape().dim_size(2);

  for (int i = 0; i < input.sentence_size(); ++i){
    // Get rid of <s> and </s>.
    int n = num_tokens_vec(i) - 2;

    TaggedSentence *sentence = result->add_sentence();
    for (int j = 0; j < n; ++j) {
      float max_score = -std::numeric_limits<double>::infinity();
      int max_index = 0;
      for (int k = 0; k < num_supertags; ++k) {
        // Offset by 1 to account for <s>.
        if (scores(i,j+1,k) > max_score) {
          max_score = scores(i,j+1,k);
          max_index = k;
        }
      }
      float prune_threshold = log(1e-5) + max_score;

      TaggedToken *token = sentence->add_token();
      token->set_word(input.sentence(i).word(j));

      // Max score goes first.
      SparseValue *score = token->add_score();
      score->set_index(max_index);
      score->set_value(max_score);

      // Adding the remaining supertags that score above the threshold.
      for (int k = 0; k < num_supertags; ++k) {
        if (scores(i,j+1,k) > prune_threshold && k != max_index) {
          SparseValue *score = token->add_score();
          score->set_index(k);
          score->set_value(scores(i,j+1,k));
        }
      }
    }
  }
}

int get_bucket(int sentence_length) {
  if (sentence_length == 0) {
    return 0;
  }
  if (sentence_length > kMaxTokens) {
    return -1;
  }
  return (sentence_length - 1) / kBucketSize;
}

JNIEXPORT jlong JNICALL Java_edu_uw_Taggerflow_initialize(JNIEnv* env, jclass clazz, jstring model, jstring spaces) {
  GraphDef graph_def;
  Status  status;

  const char* model_cstr = env->GetStringUTFChars(model, NULL);

  TF_CHECK_OK(ReadBinaryProto(Env::Default(), model_cstr, &graph_def));

  MetaSession *meta_session = new MetaSession();

  // Add the graph to the session
  SessionOptions options;
  options.config.set_allow_soft_placement(true);
  meta_session->session.reset(NewSession(options));
  TF_CHECK_OK(meta_session->session->Create(graph_def));

  // Clear the proto to save memory space.
  graph_def.Clear();

  std::cerr << "Tensorflow graph loaded from: " << model_cstr << std::endl;

  std::string spaces_str = std::string(env->GetStringUTFChars(spaces, NULL));
  read_feature_map(spaces_str, "words.txt", &meta_session->feature_maps[0]);
  read_feature_map(spaces_str, "prefix_1.txt", &meta_session->feature_maps[1]);
  read_feature_map(spaces_str, "prefix_2.txt", &meta_session->feature_maps[2]);
  read_feature_map(spaces_str, "prefix_3.txt", &meta_session->feature_maps[3]);
  read_feature_map(spaces_str, "prefix_4.txt", &meta_session->feature_maps[4]);
  read_feature_map(spaces_str, "suffix_1.txt", &meta_session->feature_maps[5]);
  read_feature_map(spaces_str, "suffix_2.txt", &meta_session->feature_maps[6]);
  read_feature_map(spaces_str, "suffix_3.txt", &meta_session->feature_maps[7]);
  read_feature_map(spaces_str, "suffix_4.txt", &meta_session->feature_maps[8]);

  // Setup commonly used indexes.
  meta_session->unknown_indexes[0] = meta_session->feature_maps[0].at("*unknown*");
  for (int i = 1; i < kNumFeatures; ++i) {
    meta_session->unknown_indexes[i] = meta_session->feature_maps[i].at("*UNKNOWN*");
    std::unordered_map<std::string,int>::const_iterator it = meta_session->feature_maps[i].find("*OOR*");
    if (it != meta_session->feature_maps[i].end()) {
      meta_session->oor_indexes[i] = it->second;
    }
  }
  for (int i = 0; i < kNumFeatures; ++i) {
    meta_session->start_indexes[i] = meta_session->feature_maps[i].at("<s>");
    meta_session->end_indexes[i] = meta_session->feature_maps[i].at("</s>");
  }

  return reinterpret_cast<long>(meta_session);
}

JNIEXPORT void JNICALL Java_edu_uw_Taggerflow_close(JNIEnv *env, jclass clazz, jlong meta_session_long) {
  MetaSession *meta_session = reinterpret_cast<MetaSession*>(meta_session_long);
  TF_CHECK_OK(meta_session->session->Close());
  delete meta_session;
  std::cerr << "Tensorflow session closed." << std::endl;
}

JNIEXPORT jbyteArray JNICALL Java_edu_uw_Taggerflow_predictPacked___3BJ(JNIEnv *env, jclass clazz, jbyteArray buffer, jlong meta_session_long) {
  pb::Arena arena;
  TaggingInput *input = pb::Arena::CreateMessage<TaggingInput>(&arena);
  jbytes_to_message(buffer, input, env);

  TaggingResult *result = pb::Arena::CreateMessage<TaggingResult>(&arena);
  parse_input(*input, reinterpret_cast<MetaSession*>(meta_session_long), result);
  return message_to_jbytes(*result, env);
}

JNIEXPORT jbyteArray JNICALL Java_edu_uw_Taggerflow_predictPacked__Ljava_lang_String_2IJ(JNIEnv* env, jclass clazz, jstring filename, jint max_batch_size, jlong meta_session_long) {
  const char* filename_cstr = env->GetStringUTFChars(filename, NULL);
  pb::Arena arena;

  std::vector<TaggingInput*> buckets;
  for (int i = 0; i < kNumBuckets; ++i) {
    buckets.push_back(pb::Arena::CreateMessage<TaggingInput>(&arena));
  }

  MetaSession *meta_session = reinterpret_cast<MetaSession*>(meta_session_long);

  std::ifstream file(filename_cstr);
  std::string line, buf;
  TaggingResult *result = pb::Arena::CreateMessage<TaggingResult>(&arena);
  TaggingInputSentence *sentence = pb::Arena::CreateMessage<TaggingInputSentence>(&arena);
  while (std::getline(file, line)) {
    sentence->Clear();
    std::stringstream ss(line);
    while (ss >> buf) {
      sentence->add_word(buf);
    }
    int bucket_index = get_bucket(sentence->word_size());
    if (bucket_index >= 0) {
      TaggingInput* bucket = buckets.at(bucket_index);
      if (bucket->sentence_size() >= max_batch_size) {
        parse_input(*bucket, meta_session, result);
        bucket->Clear();
      }
      bucket->add_sentence()->Swap(sentence);
    }
  }

  env->ReleaseStringUTFChars(filename, filename_cstr);

  TaggingInput *merged = buckets[0];
  for (unsigned i = 1; i < buckets.size(); ++i) {
    for (TaggingInputSentence &sentence : *(buckets[i]->mutable_sentence())) {
      if (merged->sentence_size() >= max_batch_size) {
        parse_input(*merged, meta_session, result);
        merged->Clear();
      }
      merged->add_sentence()->Swap(&sentence);
    }
  }
  if (merged->sentence_size() > 0) {
    parse_input(*merged, meta_session, result);
  }
  return message_to_jbytes(*result, env);
}
