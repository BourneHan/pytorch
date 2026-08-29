#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <cstring>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace c10 {

struct OperatorName final {
  std::string overload_name;
  OperatorName(std::string name, std::string overload_name)
      : overload_name(std::move(overload_name)),
        name_(std::move(name)),
        namespace_sep_(name_.find("::")) {}

  friend bool operator==(const OperatorName&, const OperatorName&) = default;

  // Fully-qualified name, e.g. "aten::add".
  const std::string& name() const {
    return name_;
  }

  // Return the namespace of this OperatorName, if it exists.  The
  // returned string_view is only live as long as the OperatorName
  // exists and name is not mutated
  std::optional<std::string_view> getNamespace() const {
    if (namespace_sep_ == std::string::npos) {
      return std::nullopt;
    } else {
      return std::string_view(name_.data(), namespace_sep_);
    }
  }

  // Part of name after the namespace prefix, e.g. "add" for "aten::add".
  std::string_view getBaseName() const {
    if (namespace_sep_ == std::string::npos) {
      return name_;
    } else {
      return std::string_view(name_).substr(namespace_sep_ + 2);
    }
  }

  // Returns true if we successfully set the namespace
  bool setNamespaceIfNotSet(const char* ns) {
    if (!getNamespace().has_value()) {
      const auto ns_len = strlen(ns);
      const auto old_name_size = name_.size();
      name_.resize(ns_len + 2 + old_name_size);
      // Shift current value of name to the end of the new space.
      auto name_data = name_.data();
      std::char_traits<char>::move(
          name_data + ns_len + 2, name_data, old_name_size);
      std::char_traits<char>::copy(name_data, ns, ns_len);
      name_[ns_len] = ':';
      name_[ns_len + 1] = ':';
      namespace_sep_ = ns_len;
      return true;
    } else {
      return false;
    }
  }

 private:
  // Private so namespace_sep_ can only go stale via the constructor or
  // setNamespaceIfNotSet, both of which keep it in sync.
  std::string name_;
  // Cached byte offset of "::" in name_ (or npos), to avoid rescanning on
  // every getNamespace()/getBaseName() call.
  std::string::size_type namespace_sep_;
};

// Non-owning view of an OperatorName.  Unlike OperatorName, most of
// its functions are constexpr, so it can be used for compile time
// computations
struct OperatorNameView final {
  std::string_view name;
  std::string_view overload_name;
  constexpr OperatorNameView(
      std::string_view name,
      std::string_view overload_name)
      : name(name), overload_name(overload_name) {}
  // Parses strings like "foo.overload" and also "foo"
  constexpr static OperatorNameView parse(std::string_view full_name) {
    auto i = full_name.find('.');
    if (i == std::string_view::npos) {
      return OperatorNameView(full_name, std::string_view());
    } else {
      return OperatorNameView(full_name.substr(0, i), full_name.substr(i + 1));
    }
  }
};

TORCH_API std::string toString(const OperatorName& opName);
TORCH_API std::ostream& operator<<(std::ostream& /*os*/, const OperatorName& /*opName*/);

} // namespace c10

namespace std {
template <>
struct hash<::c10::OperatorName> {
  size_t operator()(const ::c10::OperatorName& x) const noexcept {
    return std::hash<std::string>()(x.name()) ^
        (~std::hash<std::string>()(x.overload_name));
  }
};
} // namespace std
